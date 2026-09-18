"""本地项目库接口：/api/v1/library

面向前端「本地项目库」面板的能力：
  GET    /scan     扫描配置好的文件夹，按语言自动分类（不依赖大模型）
  GET    /file     读取某个文件的代码内容，供直接送进 AI 检测/注释/改错
  PUT    /file     覆盖保存（在线编辑保存 / 用本机文件替换）
  DELETE /file     删除（默认移进 .trash 回收站，可找回）
  GET    /status   项目库配置概览，告诉学生代码该放在哪个文件夹

设计说明：
- 读的接口都不调用大模型，因此**没配置模型时项目库依然完全可用**。
- 能扫描哪些目录只由后端配置（LIBRARY__ROOTS）决定，接口里没有任何参数
  可以指定任意路径；读/写时都会做一次「解析后仍在根目录内」的校验，
  路径穿越（../../）与指向外部的符号链接都会被拒绝。
- 写操作（PUT/DELETE）比读更危险，所以额外加了三道锁：
  1) `LIBRARY__ALLOW_WRITE=false` 可整体关掉，指向公共目录时用得上；
  2) 只能改**已存在**的源码文件，新建文件必须走「拖进项目库」的入库流程；
  3) 删除默认是移进根目录下的 `.trash/`，不是抹掉。
  PUT 还支持 `expected_sha256` 乐观锁，避免把别人刚改的内容覆盖掉。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, status

from app.api.deps import LibraryServiceDep
from app.core.trace import current_trace_id
from app.schemas.library import (
    LibraryFileContentResponse,
    LibraryFileDeleteResponse,
    LibraryFileRead,
    LibraryFileWriteRequest,
    LibraryFileWriteResponse,
    LibraryRootRead,
    LibraryScanResponse,
    LibraryStatusResponse,
)
from app.services.library_service import (
    LibraryConflictError,
    LibraryFileTooLargeError,
    LibraryPathError,
    LibraryReadOnlyError,
    LibraryScanResult,
    LibraryService,
    LibraryWriteError,
    detect_language_by_name,
    language_label,
)
from app.services.repo.language import LANGUAGE_LABELS, SUFFIX_TO_LANGUAGE

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# 结果转换
# ---------------------------------------------------------------------------
def _root_read(root) -> LibraryRootRead:
    """把服务层的根目录对象转成响应模型。"""
    return LibraryRootRead(
        name=root.name,
        path=root.path,
        exists=root.exists,
        file_count=root.file_count,
        skipped_reason=root.skipped_reason,
    )


def _scan_response(service: LibraryService, result: LibraryScanResult) -> LibraryScanResponse:
    """把扫描结果转成响应模型，并补上语言展示名。"""
    return LibraryScanResponse(
        roots=[_root_read(root) for root in result.roots],
        files=[LibraryFileRead.model_validate(item) for item in result.files],
        language_counts=result.language_counts,
        # 只给本次真的扫到的语言配展示名，前端据此生成筛选按钮
        language_labels={
            name: LANGUAGE_LABELS.get(name, name) for name in result.language_counts
        },
        total_files=result.total_files,
        total_lines=result.total_lines,
        total_bytes=result.total_bytes,
        truncated=result.truncated,
        max_file_bytes=service.settings.max_file_bytes,
        allow_write=service.settings.allow_write,
        trash_dir_name=service.settings.trash_dir_name,
        scanned_at=result.scanned_at,
        trace_id=current_trace_id(),
    )


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------
@router.get(
    "/status",
    response_model=LibraryStatusResponse,
    summary="本地项目库状态",
    description="返回当前配置的扫描根目录与各项上限，前端据此提示学生把代码放在哪里。",
)
async def library_status(service: LibraryServiceDep) -> LibraryStatusResponse:
    settings = service.settings
    # 这里刻意不做文件统计：前端加载时会调用 /scan，两份数据没必要扫两遍磁盘。
    # 状态接口只回答「目录在哪、存不存在、上限是多少」。
    roots = [
        LibraryRootRead(
            name=path.name or str(path),
            path=str(path),
            exists=path.is_dir(),
            file_count=0,
            skipped_reason="" if path.is_dir() else "目录不存在",
        )
        for path in settings.root_paths
    ]
    # 支持的语言从后缀表反推，避免两处各写一份清单而慢慢对不上
    supported = sorted(set(SUFFIX_TO_LANGUAGE.values()))
    return LibraryStatusResponse(
        roots=roots,
        default_path=str(settings.default_path),
        supported_languages={name: LANGUAGE_LABELS.get(name, name) for name in supported},
        max_files=settings.max_files,
        max_depth=settings.max_depth,
        max_file_bytes=settings.max_file_bytes,
        allow_write=settings.allow_write,
        trash_dir_name=settings.trash_dir_name,
    )


# ---------------------------------------------------------------------------
# 扫描
# ---------------------------------------------------------------------------
@router.get(
    "/scan",
    response_model=LibraryScanResponse,
    summary="扫描本地项目库并按语言分类",
    description=(
        "递归扫描配置好的文件夹，只收录能识别的源码文件（.c/.h/.java/.py 等），"
        "并按语言统计数量、行数与体积。本接口不调用大模型。"
    ),
)
async def scan_library(
    service: LibraryServiceDep,
    language: str | None = Query(
        default=None, description="只看某个语言，如 python / c / java"
    ),
    root: str | None = Query(default=None, description="只看某个根目录（按目录名）"),
    keyword: str | None = Query(default=None, description="按文件路径做包含匹配，忽略大小写"),
) -> LibraryScanResponse:
    result = service.scan(language=language, root_name=root, keyword=keyword)
    logger.info(
        "项目库扫描完成: 根目录=%s 文件=%s 语言=%s",
        [r.name for r in result.roots],
        result.total_files,
        result.language_counts,
    )
    return _scan_response(service, result)


# ---------------------------------------------------------------------------
# 读取文件内容
# ---------------------------------------------------------------------------
@router.get(
    "/file",
    response_model=LibraryFileContentResponse,
    summary="读取项目库里的一个代码文件",
    description=(
        "传入相对路径（scan 结果里的 rel_path）读取代码内容。"
        "只会读取配置根目录之内的文件，越界路径一律拒绝。"
    ),
)
async def read_library_file(
    service: LibraryServiceDep,
    path: str = Query(description="相对项目库根目录的路径，如 homework/linked_list.c"),
    root: str | None = Query(default=None, description="根目录标识，多个根目录同名文件时使用"),
) -> LibraryFileContentResponse:
    try:
        content = service.read_file(path, root)
    except LibraryPathError as exc:
        # 路径越界属于「请求不合法」，明确用 400 而不是 500，避免被误当成服务端故障
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LibraryFileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except OSError as exc:  # 文件刚好被删/被占用等
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"读取文件失败：{exc}"
        ) from exc

    language = detect_language_by_name(content.rel_path) or "unknown"
    return LibraryFileContentResponse(
        root=root or "",
        rel_path=content.rel_path,
        filename=content.rel_path.rsplit("/", 1)[-1],
        language=language,
        language_label=language_label(language),
        code=content.code,
        # 行数口径与项目库列表、代码分析结果保持一致，避免同一个文件显示不同行数
        line_count=content.line_count,
        char_count=len(content.code),
        size_bytes=content.size_bytes,
        encoding=content.encoding,
        replaced=content.replaced,
        # 乐观锁基准：前端保存时原样带回来，后端据此发现"文件被别处改过"
        sha256=content.sha256,
        modified_at=content.modified_at,
        allow_write=service.settings.allow_write,
        trace_id=current_trace_id(),
    )


# ---------------------------------------------------------------------------
# 覆盖保存（在线编辑 / 用本机文件替换）
# ---------------------------------------------------------------------------
@router.put(
    "/file",
    response_model=LibraryFileWriteResponse,
    summary="覆盖保存项目库里的一个文件",
    description=(
        "把新内容写回已存在的文件，用于「在线编辑保存」与「用本机文件替换」。"
        "只允许改配置根目录之内的源码文件；传入 expected_sha256 时做乐观锁校验，"
        "内容与打开时不一致会返回 409，避免覆盖掉别处的改动。"
    ),
)
async def write_library_file(
    service: LibraryServiceDep,
    payload: LibraryFileWriteRequest,
) -> LibraryFileWriteResponse:
    try:
        result = service.write_text(
            payload.path,
            payload.code,
            payload.root,
            expected_sha256=payload.expected_sha256,
        )
    except LibraryReadOnlyError as exc:
        # 只读模式：不是"没权限登录"，而是这个项目库被配置成不许改 → 403
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except LibraryConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except LibraryPathError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LibraryFileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except LibraryWriteError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.info(
        "项目库文件已保存: %s（%s 字节，换行符 %r）",
        result.rel_path,
        result.size_bytes,
        result.newline,
    )
    return LibraryFileWriteResponse(
        root=result.root,
        rel_path=result.rel_path,
        filename=result.filename,
        language=result.language,
        language_label=result.language_label,
        size_bytes=result.size_bytes,
        line_count=result.line_count,
        char_count=result.char_count,
        sha256=result.sha256,
        newline=result.newline,
        encoding=result.encoding,
        modified_at=result.modified_at,
        message=f"已保存 {result.rel_path}",
        trace_id=current_trace_id(),
    )


# ---------------------------------------------------------------------------
# 删除（默认移入回收站）
# ---------------------------------------------------------------------------
@router.delete(
    "/file",
    response_model=LibraryFileDeleteResponse,
    summary="删除项目库里的一个文件",
    description=(
        "默认把文件移到它所在根目录的 `.trash/` 下（带时间戳，可手动找回），"
        "传 permanent=true 才是彻底删除。删除只能作用于配置根目录之内的文件。"
    ),
)
async def delete_library_file(
    service: LibraryServiceDep,
    path: str = Query(description="相对项目库根目录的路径，如 homework/linked_list.c"),
    root: str | None = Query(default=None, description="根目录标识，多个根目录同名文件时使用"),
    permanent: bool = Query(default=False, description="true=彻底删除，false=移入回收站"),
) -> LibraryFileDeleteResponse:
    try:
        result = service.delete_file(path, root, permanent=permanent)
    except LibraryReadOnlyError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except LibraryPathError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LibraryWriteError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"删除失败：{exc}"
        ) from exc

    logger.info(
        "项目库文件已删除: %s（彻底删除=%s）", result.rel_path, result.permanent
    )
    message = (
        f"已彻底删除 {result.rel_path}"
        if result.permanent
        else f"已把 {result.rel_path} 移到回收站，需要的话可以去 {result.trash_dir} 找回"
    )
    return LibraryFileDeleteResponse(
        root=result.root,
        rel_path=result.rel_path,
        filename=result.filename,
        size_bytes=result.size_bytes,
        permanent=result.permanent,
        trash_path=result.trash_path,
        trash_dir=result.trash_dir,
        message=message,
        trace_id=current_trace_id(),
    )
