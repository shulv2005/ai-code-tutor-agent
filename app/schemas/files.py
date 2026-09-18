"""本地代码文件分类模块的 API 契约。"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# 扫描请求
# ---------------------------------------------------------------------------
class FileScanRequest(BaseModel):
    """扫描并分类的请求体。所有字段都有默认值，不传就按后端配置执行。"""

    directory: str = Field(
        default="",
        description=(
            "要扫描的子目录（相对分类根目录，如 homework）。"
            "留空表示扫描整个根目录。只接受根目录及其子目录，越界路径返回 400。"
        ),
    )
    dry_run: bool = Field(
        default=False,
        description=(
            "预演模式：只返回「打算把哪些文件移到哪里」，不移动任何文件、不写数据库。"
            "不确定目录里有什么时建议先跑一次预演。"
        ),
    )
    move_unknown: bool | None = Field(
        default=None,
        description=(
            "未知类型的文件是否也归档到 unknown/。"
            "不传则用后端配置（默认否：只登记、不移动）。"
        ),
    )
    recursive: bool | None = Field(
        default=None, description="是否递归子目录。不传则用后端配置（默认是）。"
    )
    max_files: int | None = Field(
        default=None, ge=1, le=100_000, description="本次最多处理多少个文件，不传则用后端配置。"
    )


# ---------------------------------------------------------------------------
# 上传入库
# ---------------------------------------------------------------------------
class FileUploadResponse(BaseModel):
    """把拖进来的文件保存到项目库的结果。"""

    filename: str = Field(description="实际写入的文件名；重名被改名时与上传名不同")
    rel_path: str = Field(description="相对分类根目录的路径，如 homework/main.py")
    size_bytes: int
    language: str
    language_label: str = Field(description="展示名：C / Java / Python")
    renamed: bool = Field(
        default=False, description="是否因为重名（内容不同）而另存为新文件名"
    )
    note: str = Field(default="", description="给学生的说明，例如「已另存为 main_1.c」")
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# 扫描结果
# ---------------------------------------------------------------------------
class ClassifiedFileResult(BaseModel):
    """单个文件的分类结果。"""

    filename: str
    language: str = Field(description="c / java / python / unknown")
    language_label: str = Field(description="面向展示的语言名：C / Java / Python / 未知")
    source_path: str = Field(description="扫描时的相对路径，如 homework/main.c")
    target_path: str = Field(description="处理后的相对路径；未移动时与 source_path 相同")
    size_bytes: int
    modified_at: datetime | None = None
    action: str = Field(
        description="moved=已归档 / planned=预演将归档 / kept=保留原位 / skipped=未处理"
    )
    note: str = Field(default="", description="为什么这么处理（重名改名、超大跳过等）")


class FileScanResponse(BaseModel):
    """一次扫描分类的完整结果。"""

    root: str = Field(description="分类根目录的绝对路径")
    dry_run: bool
    total: int = Field(description="本次参与分类的文件数")
    moved: int = Field(description="已归档（预演时为「将归档」）的文件数")
    unknown: int = Field(description="识别为未知类型的文件数")
    skipped: int = Field(description="因软链接/超大/读不到而跳过的文件数")
    counts: dict[str, int] = Field(default_factory=dict, description="语言 -> 文件数")
    language_labels: dict[str, str] = Field(
        default_factory=dict, description="语言 -> 展示名，供前端生成筛选项"
    )
    files: list[ClassifiedFileResult] = Field(default_factory=list)
    inserted: int = Field(default=0, description="本次新写入数据库的记录数（预演时恒为 0）")
    updated: int = Field(default=0, description="本次更新已有记录的条数（预演时恒为 0）")
    truncated: bool = Field(default=False, description="文件数是否达到上限而提前停止")
    duration_ms: float = 0.0
    scanned_at: datetime
    trace_id: str | None = None


# ---------------------------------------------------------------------------
# 列表
# ---------------------------------------------------------------------------
class ClassifiedFileRead(BaseModel):
    """数据库里的一条分类记录。"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    language: str
    language_label: str = Field(description="由 language 派生的展示名")
    path: str = Field(description="归档后的相对路径")
    absolute_path: str
    size_bytes: int
    archived: bool
    note: str | None = None
    file_modified_at: datetime | None = None
    created_at: datetime = Field(description="首次入库时间，即接口里说的「上传时间」")
    updated_at: datetime = Field(description="最近一次扫描到它的时间")
    exists: bool = Field(description="文件现在是否还在磁盘上（可能被学生在外面删掉/移走）")


class FileListResponse(BaseModel):
    """按语言列出的文件清单。"""

    total: int = Field(description="符合筛选条件的记录总数")
    language: str | None = Field(default=None, description="当前筛选的语言；null 表示全部")
    by_language: dict[str, int] = Field(
        default_factory=dict, description="各语言的记录数（不受当前筛选影响，方便前端做切换按钮）"
    )
    language_labels: dict[str, str] = Field(default_factory=dict)
    items: list[ClassifiedFileRead] = Field(default_factory=list)
