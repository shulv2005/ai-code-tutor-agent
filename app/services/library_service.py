"""本地项目库：扫描学生电脑上的代码文件夹，并按语言自动分类。

面向的场景很具体：同学一学期下来攒了一堆 .c / .java / .py 作业，散在各个
文件夹里。让ta们自己想起来"上次那个链表作业放哪儿了"很费劲，所以这里做四件事：

1. **扫描**：递归遍历配置好的根目录，找出所有能识别的源码文件；
2. **分类**：按扩展名判定语言（C / Java / Python / ...），并统计每个语言有多少
   个文件、多少行、多大；
3. **安全读取**：按「根目录 + 相对路径」读取文件内容，拒绝任何越界访问；
4. **安全写入**：在线编辑保存 / 用本机文件替换 / 删除，同样只能落在根目录之内。

安全设计（本模块最需要小心的地方）：
- 可扫描的根目录**只来自配置**，接口无法指定任意路径；
- 每次读/写都把「根目录」和「目标文件」都 `resolve()` 之后再比较，
  这样 `..\\..\\Windows\\win.ini` 和指向外部文件的符号链接都会被挡住；
- 单文件大小、文件总数、递归深度都有硬上限，避免一次误操作把整个磁盘扫穿；
- 写入类操作还有三道额外保险：
  * 只能改**已存在**的文件（新建文件必须走「拖进项目库」的入库流程）；
  * 删除默认移进根目录下的 `.trash/`，不是抹掉，学生手滑还能自己找回；
  * `LIBRARY__ALLOW_WRITE=false` 可整体关掉写入，公共目录共享时用得上。
"""

from __future__ import annotations

import codecs
import hashlib
import logging
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from app.core.config import LibrarySettings, get_settings
from app.services.repo.language import (
    LANGUAGE_LABELS,
    SUFFIX_TO_LANGUAGE,
)
from app.services.text_utils import count_text_lines

logger = logging.getLogger(__name__)

# 读取文件内容时依次尝试的编码。
# 顺序是有讲究的：学生机上的中文注释常见 utf-8 / GBK 两种，
# latin-1 放在最后兜底，保证「任何字节都能解出来」，不会因为一个奇怪字符就读不了文件。
# UTF-8 BOM 单独判断（见 read_text），因为 utf-8-sig 解码器对「无 BOM 的普通
# UTF-8 文件」同样会成功，混在候选列表里会让所有文件都被报成 utf-8-sig。
ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8", "gbk", "latin-1")

# 「其他」这一类：能识别扩展名但不在主打语言之列时归到这里
OTHER_LANGUAGE = "other"

# 写文件时的临时后缀。先写同目录的临时文件再 os.replace，
# 这样中途断电/写失败也不会留下一个被截断的半个文件（学生作业最怕这个）。
TEMP_SUFFIX = ".dsh-tmp"


class LibraryPathError(ValueError):
    """路径越界或非法。调用方应把它转成 400，而不是 500。"""


class LibraryFileTooLargeError(ValueError):
    """文件超过配置的大小上限。"""


class LibraryWriteError(ValueError):
    """写入类操作失败（只读模式、内容是二进制、磁盘错误等）。"""


class LibraryReadOnlyError(LibraryWriteError):
    """项目库被配置成只读（LIBRARY__ALLOW_WRITE=false）。"""


class LibraryConflictError(LibraryWriteError):
    """乐观锁冲突：文件在别处被改过，不能盲目覆盖。"""


@dataclass(slots=True)
class LibraryFile:
    """项目库里收录的一个代码文件。"""

    root: str          # 所属根目录的标识（展示用）
    rel_path: str      # 相对根目录的路径，前端点击时把它传回来
    filename: str      # 文件名
    language: str      # 语言标识，如 c / java / python
    language_label: str  # 展示名，如 C / Java / Python
    size_bytes: int
    line_count: int | None  # 超过大小上限时为 None（只登记元信息，没读内容）
    modified_at: datetime | None
    too_large: bool = False


@dataclass(slots=True)
class LibraryRoot:
    """一个被扫描的根目录。"""

    name: str
    path: str
    exists: bool
    file_count: int = 0
    skipped_reason: str = ""


@dataclass(slots=True)
class LibraryScanResult:
    """一次扫描的完整结果。"""

    roots: list[LibraryRoot] = field(default_factory=list)
    files: list[LibraryFile] = field(default_factory=list)
    language_counts: dict[str, int] = field(default_factory=dict)
    total_files: int = 0
    total_lines: int = 0
    total_bytes: int = 0
    truncated: bool = False  # 是否因为撞到 max_files 上限而提前停止
    scanned_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True)
class LibraryTextContent:
    """读出来的一个文件：内容 + 元信息。

    `sha256` 是给「在线编辑保存」用的乐观锁基准值：
    前端打开文件时记下它，保存时原样带回来，后端再算一次对比。
    对不上就说明文件在别处被改过（比如学生自己用记事本又存了一次），
    这时宁可让ta重新打开，也不能默默把人家的改动盖掉。
    """

    code: str
    rel_path: str
    encoding: str
    replaced: bool          # 是否出现过无法解码的字符（已用替换字符顶上）
    size_bytes: int
    sha256: str
    line_count: int
    modified_at: datetime | None


@dataclass(slots=True)
class LibraryWriteResult:
    """一次写入（在线编辑保存 / 替换）的结果。"""

    root: str
    rel_path: str
    filename: str
    language: str
    language_label: str
    size_bytes: int
    line_count: int
    char_count: int
    sha256: str              # 写入后的新指纹，前端用它更新乐观锁基准
    newline: str             # 实际使用的换行符（保留原文件的习惯）
    encoding: str = "utf-8"
    modified_at: datetime | None = None


@dataclass(slots=True)
class LibraryDeleteResult:
    """一次删除的结果。"""

    root: str
    rel_path: str
    filename: str
    size_bytes: int
    permanent: bool          # True = 彻底删除；False = 移进了回收站
    trash_path: str = ""     # 移到回收站时的绝对路径，前端展示给用户去手动找回
    trash_dir: str = ""      # 回收站目录，方便前端提示"文件去哪了"


class LibraryService:
    """本地项目库服务：无状态，配置在构造时注入。"""

    def __init__(self, settings: LibrarySettings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls) -> LibraryService:
        """按全局配置构造服务实例。"""
        return cls(get_settings().library)

    # ------------------------------------------------------------------
    # 扫描
    # ------------------------------------------------------------------
    def scan(
        self,
        *,
        language: str | None = None,
        root_name: str | None = None,
        keyword: str | None = None,
    ) -> LibraryScanResult:
        """遍历所有根目录，返回分类后的文件清单。

        Args:
            language: 只看某个语言（如 "python"），None 表示全部。
            root_name: 只看某个根目录（根目录标识），None 表示全部。
            keyword: 按文件名/相对路径做包含匹配（忽略大小写）。
        """
        result = LibraryScanResult()
        wanted = language.lower() if language else None
        needle = keyword.lower() if keyword else None

        for path in self.settings.root_paths:
            root = LibraryRoot(name=path.name or str(path), path=str(path), exists=path.is_dir())
            if not root.exists:
                root.skipped_reason = "目录不存在"
                result.roots.append(root)
                continue

            if root_name and root.name != root_name:
                result.roots.append(root)
                continue

            for item in self._walk(path):
                if wanted and item.language != wanted:
                    continue
                if needle and needle not in item.rel_path.lower():
                    continue
                result.files.append(item)
                root.file_count += 1
                if len(result.files) >= self.settings.max_files:
                    result.truncated = True
                    break
            result.roots.append(root)
            if result.truncated:
                # 已经到上限就没必要继续扫后面的根目录了
                logger.warning("项目库扫描达到上限 %s 个文件，已提前停止", self.settings.max_files)
                break

        self._fill_stats(result)
        return result

    def _walk(self, root: Path):
        """深度优先遍历一个根目录，产出代码文件（生成器，边扫边返回）。"""
        base_depth = len(root.parts)
        # 用显式栈而不是递归：目录层级深时不会爆栈，也便于控制深度
        stack: list[Path] = [root]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(os.scandir(current), key=lambda e: e.name.lower())
            except (PermissionError, OSError) as exc:
                # 某个子目录没权限不应该让整次扫描失败，记日志跳过即可
                logger.debug("跳过无法读取的目录 %s: %s", current, exc)
                continue

            for entry in entries:
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue

                if is_dir:
                    if entry.name in self.settings.excluded_dirs:
                        continue
                    child = Path(entry.path)
                    if len(child.parts) - base_depth >= self.settings.max_depth:
                        continue
                    stack.append(child)
                    continue

                item = self._build_file(root, Path(entry.path))
                if item is not None:
                    yield item

    def _build_file(self, root: Path, path: Path) -> LibraryFile | None:
        """把一个磁盘文件转成 LibraryFile；不是代码或读不到时返回 None。"""
        language = detect_language_by_name(path.name)
        if language is None:
            return None

        try:
            stat = path.stat()
        except OSError:
            return None

        rel = self._relative(root, path)
        too_large = stat.st_size > self.settings.max_file_bytes
        return LibraryFile(
            root=root.name or str(root),
            rel_path=rel,
            filename=path.name,
            language=language,
            language_label=language_label(language),
            size_bytes=stat.st_size,
            # 超大文件不读内容，行数留空，前端会显示「文件过大」
            line_count=None if too_large else count_lines(path),
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            too_large=too_large,
        )

    @staticmethod
    def _relative(root: Path, path: Path) -> str:
        """算出相对根目录的路径，统一用正斜杠（前端和 URL 都好处理）。"""
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def _fill_stats(result: LibraryScanResult) -> None:
        """统计各语言的文件数、总行数、总字节数。"""
        counter: Counter[str] = Counter()
        for item in result.files:
            counter[item.language] += 1
            result.total_bytes += item.size_bytes
            result.total_lines += item.line_count or 0
        result.total_files = len(result.files)
        # 数量多的语言排前面，前端筛选按钮一眼就能看出主力语言
        result.language_counts = dict(counter.most_common())

    # ------------------------------------------------------------------
    # 读取文件
    # ------------------------------------------------------------------
    def resolve_file(self, rel_path: str, root_name: str | None = None) -> Path:
        """把「根目录 + 相对路径」解析成真实路径，并校验没有越界。

        Raises:
            LibraryPathError: 路径为空、含绝对路径、越界、或文件不存在。
        """
        cleaned = (rel_path or "").strip().replace("\\", "/")
        if not cleaned:
            raise LibraryPathError("文件路径不能为空")
        if cleaned.startswith("/") or (len(cleaned) > 1 and cleaned[1] == ":"):
            raise LibraryPathError("只接受相对于项目库根目录的路径")
        # 显式拦掉 ..：虽然后面的 resolve 比较也能拦住，但提前拒绝能给出更清楚的提示
        if any(part == ".." for part in cleaned.split("/")):
            raise LibraryPathError("路径中不允许出现 ..")

        candidates = self.settings.root_paths
        if root_name:
            selected = [p for p in candidates if (p.name or str(p)) == root_name]
            if not selected:
                raise LibraryPathError(f"没有名为 {root_name} 的项目库根目录")
            candidates = selected

        for root in candidates:
            try:
                root_real = root.resolve(strict=True)
            except (OSError, FileNotFoundError):
                continue
            target = (root_real / cleaned).resolve()
            # resolve() 会把符号链接展开，所以这一步同时挡住了「软链接指向外部」
            if not target.is_relative_to(root_real):
                continue
            if not target.is_file():
                continue
            return target

        raise LibraryPathError(f"文件不存在或不在项目库范围内：{cleaned}")

    def read_text(
        self, rel_path: str, root_name: str | None = None
    ) -> tuple[str, str, str, bool]:
        """读取文件内容（兼容旧签名）。

        Returns:
            (代码文本, 相对路径, 使用的编码, 是否被截断)
        """
        content = self.read_file(rel_path, root_name)
        return content.code, content.rel_path, content.encoding, content.replaced

    def read_file(self, rel_path: str, root_name: str | None = None) -> LibraryTextContent:
        """读取文件内容，并带上大小 / 指纹 / 行数等元信息。

        与 `read_text` 的区别只是"多给了几个字段"：接口层需要 sha256 做乐观锁，
        又不想为了拿它把文件读两遍。
        """
        path = self.resolve_file(rel_path, root_name)
        size = path.stat().st_size
        if size > self.settings.max_file_bytes:
            raise LibraryFileTooLargeError(
                f"文件 {path.name} 有 {size // 1024} KB，"
                f"超过上限 {self.settings.max_file_bytes // 1024} KB，请换一个小一点的文件"
            )

        raw = path.read_bytes()
        encoding = "utf-8"
        replaced = False
        # Windows 记事本另存为 UTF-8 时会加 BOM，必须先把 BOM 去掉再交给下游：
        # 否则代码第一个字符会是一个看不见的 \ufeff，贴给 AI 或写回文件都不干净。
        if raw.startswith(codecs.BOM_UTF8):
            code, encoding = raw.decode("utf-8-sig"), "utf-8-sig"
        else:
            code = ""
            for candidate in ENCODING_CANDIDATES:
                try:
                    code, encoding = raw.decode(candidate), candidate
                    break
                except UnicodeDecodeError:
                    continue
            if not code:
                # latin-1 理论上不会失败；真到了这里说明文件不是文本，用替换字符兜底
                code, encoding, replaced = raw.decode("utf-8", errors="replace"), "utf-8", True

        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            mtime = None

        return LibraryTextContent(
            code=code,
            rel_path=self._relative_for(path),
            encoding=encoding,
            replaced=replaced,
            size_bytes=size,
            sha256=hashlib.sha256(raw).hexdigest(),
            line_count=count_text_lines(code),
            modified_at=mtime,
        )

    def _relative_for(self, path: Path) -> str:
        """给一个已解析的绝对路径，找出它相对哪个根目录。"""
        root = self._root_for(path)
        if root is not None:
            return path.relative_to(root).as_posix()
        return path.name

    def _root_for(self, path: Path) -> Path | None:
        """找出一个已解析的绝对路径属于哪个根目录；不属于任何根目录时返回 None。

        写入类操作（保存/删除）用它来确认"这个文件确实在项目库里"，
        并且知道该把回收站建在哪个根目录下。
        """
        for root in self.settings.root_paths:
            try:
                root_real = root.resolve(strict=True)
            except OSError:
                continue
            if path.is_relative_to(root_real):
                return root_real
        return None

    # ------------------------------------------------------------------
    # 写入文件（在线编辑保存 / 用本机文件替换）
    # ------------------------------------------------------------------
    def _require_writable(self) -> None:
        """检查是否允许写入。只读模式下所有写操作都要在这里被拦下。"""
        if not self.settings.allow_write:
            raise LibraryReadOnlyError(
                "项目库当前是只读模式（LIBRARY__ALLOW_WRITE=false），"
                "编辑 / 替换 / 删除都已被禁用"
            )

    def write_text(
        self,
        rel_path: str,
        code: str,
        root_name: str | None = None,
        expected_sha256: str | None = None,
    ) -> LibraryWriteResult:
        """把内容覆盖写回项目库里的某个文件。

        这是「在线编辑保存」和「用本机文件替换」共用的底层实现，两者只有
        内容来源不同，落盘逻辑完全一样。

        Args:
            rel_path: 相对根目录的路径（scan 结果里的 rel_path）。
            code: 新的完整内容。
            root_name: 多个根目录同名文件时用来消歧。
            expected_sha256: 可选。前端打开文件时拿到的指纹；给了就做乐观锁校验，
                与磁盘上的当前内容对不上时抛 LibraryConflictError，避免把
                别人（或学生自己在别的编辑器里）的改动悄悄盖掉。

        Raises:
            LibraryReadOnlyError: 只读模式。
            LibraryPathError: 路径越界、或文件不存在（新建请走入库流程）。
            LibraryFileTooLargeError: 内容超过单文件上限。
            LibraryWriteError: 内容含二进制字符、或磁盘写入失败。
            LibraryConflictError: 乐观锁冲突。
        """
        self._require_writable()

        if not isinstance(code, str):
            raise LibraryWriteError("要写入的内容必须是文本")

        # 学生偶尔会把压缩包/图片误当作代码替换进来：里面有 NUL 字节就明确拒绝，
        # 免得把一个文本文件写成"二进制垃圾"，之后连打开都失败。
        if "\x00" in code:
            raise LibraryWriteError("内容里含有二进制字符（NUL），不敢写进代码文件")

        payload = code.encode("utf-8")
        if len(payload) > self.settings.max_file_bytes:
            raise LibraryFileTooLargeError(
                f"内容有 {len(payload) // 1024} KB，超过单文件上限 "
                f"{self.settings.max_file_bytes // 1024} KB"
            )

        path = self.resolve_file(rel_path, root_name)

        # 乐观锁：前端带回来的指纹必须和磁盘现状一致
        if expected_sha256:
            current = hashlib.sha256(path.read_bytes()).hexdigest()
            if current != expected_sha256.strip().lower():
                raise LibraryConflictError(
                    "这个文件在别处被改动过（内容和打开时不一样），"
                    "为避免盖掉新内容，请重新打开文件再改"
                )

        # 保留原文件的换行习惯：Windows 上大多是 CRLF，
        # 如果统一写成 LF，学生用记事本打开会发现"整个文件都变了"。
        original = path.read_bytes()
        newline = "\r\n" if b"\r\n" in original else "\n"
        text = code.replace("\r\n", "\n").replace("\r", "\n")
        if newline == "\r\n":
            text = text.replace("\n", "\r\n")
        payload = text.encode("utf-8")

        # 先写同目录的临时文件，再原子替换：
        # 中途失败时原文件仍然完好，不会出现被截断的半个作业。
        temp_path = path.with_name(path.name + TEMP_SUFFIX)
        try:
            temp_path.write_bytes(payload)
            os.replace(temp_path, path)
        except OSError as exc:
            temp_path.unlink(missing_ok=True)
            raise LibraryWriteError(f"写入失败：{exc}") from exc

        language = detect_language_by_name(path.name) or OTHER_LANGUAGE
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        except OSError:
            mtime = None

        logger.info("项目库文件已保存: %s（%s 字节）", path, len(payload))
        root = self._root_for(path)
        return LibraryWriteResult(
            root=(root.name if root else "") or "",
            rel_path=self._relative_for(path),
            filename=path.name,
            language=language,
            language_label=language_label(language),
            size_bytes=len(payload),
            line_count=count_text_lines(code),
            char_count=len(code),
            sha256=hashlib.sha256(payload).hexdigest(),
            newline=newline,
            modified_at=mtime,
        )

    # ------------------------------------------------------------------
    # 删除文件
    # ------------------------------------------------------------------
    def delete_file(
        self,
        rel_path: str,
        root_name: str | None = None,
        permanent: bool = False,
    ) -> LibraryDeleteResult:
        """删除项目库里的一个文件。

        默认**不是真删**：文件会被移动到自己根目录下的 `.trash/` 里，
        名字带时间戳（`20260218-153000__homework__main.c`），学生手滑点错了
        还能自己进去找回来。传 `permanent=True` 才是彻底删除。

        Raises:
            LibraryReadOnlyError: 只读模式。
            LibraryPathError: 路径越界或文件不存在。
            LibraryWriteError: 移动/删除失败。
        """
        self._require_writable()

        path = self.resolve_file(rel_path, root_name)
        size = path.stat().st_size
        root = self._root_for(path)
        rel = self._relative_for(path)

        # 彻底删除，或没配回收站目录：直接删
        trash_name = (self.settings.trash_dir_name or "").strip()
        if permanent or not trash_name:
            try:
                path.unlink()
            except OSError as exc:
                raise LibraryWriteError(f"删除失败：{exc}") from exc
            logger.info("项目库文件已彻底删除: %s", path)
            return LibraryDeleteResult(
                root=(root.name if root else "") or "",
                rel_path=rel,
                filename=path.name,
                size_bytes=size,
                permanent=True,
            )

        if root is None:
            # 理论上 resolve_file 已经保证了这一点，这里只是不给"删到根目录外"留任何缝
            raise LibraryPathError(f"文件不在项目库范围内，拒绝删除：{rel}")

        trash_dir = root / trash_name
        try:
            trash_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            flat = rel.replace("/", "__")
            target = _unique_path(trash_dir / f"{stamp}__{flat}")
            shutil.move(str(path), str(target))
        except OSError as exc:
            raise LibraryWriteError(f"移动到回收站失败：{exc}") from exc

        logger.info("项目库文件已移入回收站: %s -> %s", path, target)
        return LibraryDeleteResult(
            root=(root.name if root else "") or "",
            rel_path=rel,
            filename=path.name,
            size_bytes=size,
            permanent=False,
            trash_path=str(target),
            trash_dir=str(trash_dir),
        )


# ---------------------------------------------------------------------------
# 模块级工具函数
# ---------------------------------------------------------------------------
def detect_language_by_name(filename: str) -> str | None:
    """按扩展名判断语言；不认识的扩展名返回 None。

    注意要取「最长的后缀」而不是「第一个命中的后缀」：
    `.h` 和 `.hh` 都能匹配到 `xxx.hh` 吗？不能——但反过来，
    如果后缀表里同时有 `.h` 和 `.hpp`，用 `endswith` 逐个判断时
    顺序就很重要。这里改成先排序再匹配，保证长后缀优先。
    """
    lowered = filename.lower()
    for suffix in sorted(SUFFIX_TO_LANGUAGE, key=len, reverse=True):
        if lowered.endswith(suffix):
            return SUFFIX_TO_LANGUAGE[suffix]
    return None


def language_label(language: str) -> str:
    """把语言标识转成展示名。"""
    return LANGUAGE_LABELS.get(language, language)


def count_lines(path: Path, block_size: int = 64 * 1024) -> int:
    """分块统计文本行数，避免为了数行数把大文件整个读进内存。

    口径与 `app.services.text_utils.count_text_lines` 完全一致（末尾换行不算新行、
    空文件 0 行），区别只在于这里是流式读文件、那边是对已有字符串计数。
    两者必须给出相同结果，`tests/test_library.py` 里有专门的用例守着。
    """
    total = 0
    tail_was_newline = True
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(block_size):
                total += chunk.count(b"\n")
                tail_was_newline = chunk.endswith(b"\n")
    except OSError:
        return 0
    if total == 0:
        # 没有任何换行符：文件非空就是 1 行，空文件是 0 行
        try:
            return 0 if path.stat().st_size == 0 else 1
        except OSError:
            return 0
    return total if tail_was_newline else total + 1


def is_code_file(filename: str) -> bool:
    """判断一个文件名是不是可识别的源码文件。"""
    return detect_language_by_name(filename) is not None


def _unique_path(path: Path) -> Path:
    """目标已存在时加 `_1` `_2` 后缀，避免同名文件互相覆盖。

    典型场景：同一秒内删除了两个同名的 `main.c`（来自不同子目录），
    回收站文件名会撞车，这里给它让开。
    """
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    # 极端情况：同名文件多到 1000 个，用时间戳兜底，保证一定能落盘
    return path.with_name(f"{stem}_{datetime.now().strftime('%f')}{suffix}")


@lru_cache(maxsize=1)
def get_library_service() -> LibraryService:
    """获取项目库服务单例（配置变化后需调用 cache_clear）。"""
    return LibraryService.from_settings()


__all__ = [
    "ENCODING_CANDIDATES",
    "LibraryConflictError",
    "LibraryDeleteResult",
    "LibraryFile",
    "LibraryFileTooLargeError",
    "LibraryPathError",
    "LibraryReadOnlyError",
    "LibraryRoot",
    "LibraryScanResult",
    "LibraryService",
    "LibraryTextContent",
    "LibraryWriteError",
    "LibraryWriteResult",
    "count_lines",
    "detect_language_by_name",
    "get_library_service",
    "is_code_file",
    "language_label",
]
