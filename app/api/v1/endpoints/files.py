"""本地代码文件自动分类接口：/api/v1/files

按需求提供两个接口：
  POST /scan   扫描指定目录，按扩展名识别语言并把文件归档到 c/ java/ python/
  GET  /list   按语言列出已分类的文件（读 SQLite 台账）

设计说明：
- **扫描是真会搬文件的**，所以 `dry_run=true` 可以先预演；预演不移动、不写库。
- 能被扫描的目录只允许是后端配置的 `CLASSIFIER__ROOT` 及其子目录，
  接口里没有任何参数能指向任意路径 —— 这个服务最终是给学生双击运行的，
  一旦能指定任意路径，就等于把整台机器的文件都暴露给了一个"搬运工"。
- 目录遍历与文件移动是同步阻塞操作，放在线程池里执行（`run_in_threadpool`），
  否则目录一大就会把事件循环卡住，页面上表现为"整个服务没反应"。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from app.api.deps import DbSession, FileClassifierDep
from app.core.trace import current_trace_id
from app.schemas.files import (
    ClassifiedFileRead,
    ClassifiedFileResult,
    FileListResponse,
    FileScanRequest,
    FileScanResponse,
    FileUploadResponse,
)
from app.services.file_classifier import (
    LANGUAGE_DISPLAY,
    DuplicateFileError,
    FileClassifierError,
    FileClassifierPathError,
)
from app.services.file_record_service import FileRecordService

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# 前端拖拽入库：保存文件到项目库
# ---------------------------------------------------------------------------
@router.post(
    "/upload",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传一个代码文件到项目库",
    description=(
        "把前端拖进来的文件保存到分类根目录，**等待 `/files/scan` 按语言归档**。\n\n"
        "只接受 C / Java / Python 的源码文件（`.c` / `.h` / `.java` / `.py`）。\n\n"
        "重复判定用「同名 + 内容相同」两个条件：\n"
        "- 完全相同的文件再次上传 → **409**，detail 为「文件已存在：<路径>」；\n"
        "- 同名但内容不同（例如两个学生都交了 `main.c`）→ 自动另存为 `main_1.c`，\n"
        "  既不覆盖已有文件，也不会让学生以为自己的作业没交上。"
    ),
)
async def upload_file(
    classifier: FileClassifierDep,
    file: UploadFile = File(..., description="要入库的代码文件"),
) -> FileUploadResponse:
    """把上传的文件保存到项目库根目录，等 /files/scan 归档。"""
    try:
        content = await file.read()
        # 写盘是阻塞 IO，丢到线程池里，避免卡住事件循环
        result = await run_in_threadpool(classifier.save_upload, file.filename or "", content)
    except DuplicateFileError as exc:
        # 用 409 而不是 400：前端要能区分"文件已存在"和"参数不合法"两种提示
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except FileClassifierPathError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileClassifierError as exc:
        # 文件过大用 413，其它（空文件、写盘失败）用 400
        code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if "超过上限" in str(exc)
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail=str(exc)) from exc

    logger.info(
        "文件已入库: %s（%s 字节，%s）",
        result.relative_path,
        result.size_bytes,
        result.language,
    )
    return FileUploadResponse(
        filename=result.filename,
        rel_path=result.relative_path,
        size_bytes=result.size_bytes,
        language=result.language,
        language_label=result.language_label,
        renamed=result.renamed,
        note=result.note,
        trace_id=current_trace_id(),
    )


# ---------------------------------------------------------------------------
# 扫描并分类
# ---------------------------------------------------------------------------
@router.post(
    "/scan",
    response_model=FileScanResponse,
    summary="扫描并分类本地代码文件",
    description=(
        "递归扫描指定目录（默认是配置的分类根目录），按扩展名识别语言：\n"
        "- `.c` / `.h` → C，归档到 `c/`\n"
        "- `.java` → Java，归档到 `java/`\n"
        "- `.py` → Python，归档到 `python/`\n"
        "- 其它 → 未知，默认只登记不移动（`move_unknown=true` 时归到 `unknown/`）\n\n"
        "请求体可以整个省略（等价于用默认参数扫描根目录）。"
        "`dry_run=true` 时只返回计划、不移动文件也不写数据库。"
    ),
)
async def scan_files(
    session: DbSession,
    classifier: FileClassifierDep,
    payload: FileScanRequest | None = Body(default=None),
) -> FileScanResponse:
    """扫描目录、识别语言、归档文件，并把元信息写入 SQLite。"""
    request = payload or FileScanRequest()

    try:
        # 目录遍历 + 文件移动是阻塞 IO，丢到线程池，避免卡住事件循环
        report = await run_in_threadpool(
            classifier.classify,
            directory=request.directory,
            dry_run=request.dry_run,
            move_unknown=request.move_unknown,
            recursive=request.recursive,
            max_files=request.max_files,
        )
    except FileClassifierPathError as exc:
        # 目录越界/不存在属于"请求不合法"，用 400 让前端能区分于服务端故障
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileClassifierError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc

    inserted, updated = await FileRecordService.save_report(session, report)

    return FileScanResponse(
        root=report.root,
        dry_run=report.dry_run,
        total=report.total,
        moved=report.moved,
        unknown=report.unknown,
        skipped=report.skipped,
        counts=report.counts,
        language_labels={name: LANGUAGE_DISPLAY.get(name, name) for name in report.counts},
        files=[
            ClassifiedFileResult(
                filename=item.filename,
                language=item.language,
                language_label=item.language_label,
                source_path=item.source_path,
                target_path=item.target_path,
                size_bytes=item.size_bytes,
                modified_at=item.modified_at,
                action=item.action,
                note=item.note,
            )
            for item in report.files
        ],
        inserted=inserted,
        updated=updated,
        truncated=report.truncated,
        duration_ms=round(report.duration_ms, 3),
        scanned_at=report.scanned_at,
        trace_id=current_trace_id(),
    )


# ---------------------------------------------------------------------------
# 按语言列出文件
# ---------------------------------------------------------------------------
@router.get(
    "/list",
    response_model=FileListResponse,
    summary="按语言列出已分类的文件",
    description=(
        "从 SQLite 台账里读取分类结果，可按语言筛选（c / java / python / unknown），"
        "按入库时间倒序分页返回。每条记录带 `exists` 字段，"
        "表示文件此刻是否还在磁盘上（学生在资源管理器里删掉的情况）。"
    ),
)
async def list_files(
    session: DbSession,
    language: str | None = Query(
        default=None, description="按语言筛选：c / java / python / unknown；留空表示全部"
    ),
    limit: int = Query(default=50, ge=1, le=500, description="本页最多返回多少条"),
    offset: int = Query(default=0, ge=0, description="跳过多少条，用于翻页"),
) -> FileListResponse:
    """按语言分页列出分类记录。"""
    normalized = (language or "").strip().lower() or None
    if normalized is not None and normalized not in LANGUAGE_DISPLAY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"不支持的语言：{language}。可选值："
                + "、".join(sorted(LANGUAGE_DISPLAY))
            ),
        )

    items, total = await FileRecordService.list_files(
        session, language=normalized, limit=limit, offset=offset
    )
    by_language = await FileRecordService.count_by_language(session)

    return FileListResponse(
        total=total,
        language=normalized,
        by_language=by_language,
        language_labels={name: LANGUAGE_DISPLAY.get(name, name) for name in by_language},
        items=[
            ClassifiedFileRead(
                id=item.id,
                filename=item.filename,
                language=item.language,
                language_label=LANGUAGE_DISPLAY.get(item.language, item.language),
                path=item.path,
                absolute_path=item.absolute_path,
                size_bytes=item.size_bytes,
                archived=item.archived,
                note=item.note,
                file_modified_at=item.file_modified_at,
                created_at=item.created_at,
                updated_at=item.updated_at,
                # 只做一次存在性判断：文件可能在程序外面被删掉/移走，
                # 列表里如实标出来，比让学生对着一条不存在的记录发呆要好
                exists=_exists(item.absolute_path),
            )
            for item in items
        ],
    )


def _exists(absolute_path: str) -> bool:
    """判断记录对应的文件此刻是否还在磁盘上。"""
    from pathlib import Path

    try:
        return Path(absolute_path).is_file()
    except OSError:  # 路径过长、盘符已拔出等情况
        return False
