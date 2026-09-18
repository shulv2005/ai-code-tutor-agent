"""本地项目库模块的 API 契约。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# 扫描结果
# ---------------------------------------------------------------------------
class LibraryRootRead(BaseModel):
    """一个被扫描的根目录。"""

    name: str = Field(description="根目录标识，读取文件时用它区分同名文件")
    path: str = Field(description="宿主机上的绝对路径，展示给学生看代码放在哪")
    exists: bool
    file_count: int = 0
    skipped_reason: str = Field(default="", description="被跳过时的原因，如「目录不存在」")


class LibraryFileRead(BaseModel):
    """项目库里的一个代码文件。"""

    model_config = ConfigDict(from_attributes=True)

    root: str
    rel_path: str = Field(description="相对根目录的路径，读取内容时传回来")
    filename: str
    language: str
    language_label: str
    size_bytes: int
    line_count: int | None = Field(default=None, description="行数；文件过大时为 null")
    modified_at: datetime | None = None
    too_large: bool = Field(default=False, description="是否超过单文件大小上限")


class LibraryScanResponse(BaseModel):
    """一次扫描的完整结果，前端用它渲染文件树与筛选按钮。"""

    roots: list[LibraryRootRead] = Field(default_factory=list)
    files: list[LibraryFileRead] = Field(default_factory=list)
    language_counts: dict[str, int] = Field(
        default_factory=dict, description="语言 -> 文件数，已按数量倒序"
    )
    language_labels: dict[str, str] = Field(
        default_factory=dict, description="语言 -> 展示名，供前端生成筛选项"
    )
    total_files: int = 0
    total_lines: int = 0
    total_bytes: int = 0
    truncated: bool = Field(default=False, description="文件数是否达到上限而提前停止")
    max_file_bytes: int = 0
    allow_write: bool = Field(
        default=True, description="是否允许编辑/替换/删除；false 时前端把这些按钮禁用"
    )
    trash_dir_name: str = Field(default=".trash", description="删除时文件被移到的目录名")
    scanned_at: datetime
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# 文件内容
# ---------------------------------------------------------------------------
class LibraryFileContentResponse(BaseModel):
    """从项目库里读出来的一个文件内容。"""

    root: str
    rel_path: str
    filename: str
    language: str
    language_label: str
    code: str
    line_count: int
    char_count: int
    size_bytes: int
    encoding: str = Field(description="实际用于解码的编码，如 utf-8 / gbk")
    replaced: bool = Field(default=False, description="是否出现过无法解码的字符（已替换）")
    sha256: str = Field(
        default="",
        description="内容指纹。在线编辑保存时原样带回，用于发现「文件已被别处改动」",
    )
    modified_at: datetime | None = None
    allow_write: bool = Field(default=True, description="是否允许把这个文件改回去")
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# 写入（在线编辑保存 / 用本机文件替换）
# ---------------------------------------------------------------------------
class LibraryFileWriteRequest(BaseModel):
    """覆盖保存一个项目库文件。编辑保存与「替换」共用这个请求体。"""

    path: str = Field(
        min_length=1,
        description="相对项目库根目录的路径（scan 结果里的 rel_path）",
    )
    root: str | None = Field(
        default=None, description="根目录标识，多个根目录存在同名文件时用它消歧"
    )
    code: str = Field(description="新的完整文件内容")
    expected_sha256: str | None = Field(
        default=None,
        description=(
            "可选乐观锁：打开文件时拿到的指纹。"
            "与磁盘现状不一致时返回 409，避免覆盖掉别处的改动"
        ),
    )


class LibraryFileWriteResponse(BaseModel):
    """写入成功后的文件现状。"""

    root: str
    rel_path: str
    filename: str
    language: str
    language_label: str
    size_bytes: int
    line_count: int
    char_count: int
    sha256: str = Field(description="写入后的新指纹，前端用它更新乐观锁基准")
    newline: str = Field(default="\n", description="实际写回的换行符（保留原文件习惯）")
    encoding: str = "utf-8"
    modified_at: datetime | None = None
    message: str = Field(default="", description="给学生的提示，例如「已保存」")
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# 删除
# ---------------------------------------------------------------------------
class LibraryFileDeleteResponse(BaseModel):
    """删除结果。默认是「移入回收站」而不是真删。"""

    root: str
    rel_path: str
    filename: str
    size_bytes: int
    permanent: bool = Field(description="true=彻底删除；false=移进了回收站，还能找回")
    trash_path: str = Field(default="", description="回收站里的绝对路径")
    trash_dir: str = Field(default="", description="回收站目录，前端提示学生去哪找回")
    message: str = Field(default="")
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------
class LibraryStatusResponse(BaseModel):
    """项目库配置概览：前端据此提示学生「代码该放哪个文件夹」。"""

    roots: list[LibraryRootRead] = Field(default_factory=list)
    default_path: str = Field(description="默认项目库目录，开箱即用")
    supported_languages: dict[str, str] = Field(default_factory=dict)
    max_files: int = 0
    max_depth: int = 0
    max_file_bytes: int = 0
    allow_write: bool = Field(default=True, description="是否允许编辑/替换/删除")
    trash_dir_name: str = Field(default=".trash", description="回收站目录名")
