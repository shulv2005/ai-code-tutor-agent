"""本地代码文件自动分类（归档）服务。

面向课堂场景：同学把一学期的作业堆在 `data/library/` 里（可能还夹着几个子文件夹），
本模块负责

  1. **扫描**：递归遍历指定目录，跳过版本库、虚拟环境等无关目录；
  2. **识别**：按扩展名判断语言 —— `.c` / `.h` → C，`.java` → Java，`.py` → Python，
     其余一律记为 `unknown`（未知）；
  3. **归档**：把识别出的文件移动到对应子目录
     `data/library/c/`、`data/library/java/`、`data/library/python/`；
     未知文件默认**只登记不移动**（想一起归到 `unknown/` 就打开 `move_unknown`）；
  4. **接收前端拖拽**：`save_upload()` 把浏览器拖进来的文件存到根目录，
     再走上面的归档流程（见 `app/api/v1/endpoints/files.py` 的 `/upload`）；
  5. **产出结构化结果**：交给上层写进 SQLite（见 `app/services/file_record_service.py`）。

注意归档根目录默认与「本地项目库」同一个文件夹（`data/library`）：
学生拖进来的文件归档后依然能被项目库扫描到，列表里立刻就能看见。

------------------------------------------------------------
设计上的关键取舍（每条都是"会丢文件"的风险点，别轻易改）
------------------------------------------------------------

* **只认三种语言**。需求明确只归档 C / Java / Python。`.go`、`.js` 这类扩展名虽然
  项目里的解析器认识，但不在归档范围内，统一记成 `unknown` 并在说明里写清
  "它其实是 Go"，而不是含糊地当成"不认识的扩展名"。要支持更多语言，
  只需往 `ARCHIVE_DIRS` 里加一条。

* **可重复执行（幂等）**。根目录下的归档目录 `c/`、`java/`、`python/`、`unknown/`
  在扫描时会被跳过，所以连点两次「扫描」不会把 `c/a.c` 再移成 `c/c/a.c`，
  也不会产生重复记录。这是本模块最容易被忽略、又最容易出丑的地方。

* **绝不覆盖同名文件**。目标目录已有同名文件时自动加 `_1`、`_2` 后缀。
  两个子文件夹里各有一个 `main.c` 是很常见的情况，直接覆盖等于删掉同学的作业。

* **软链接一律跳过**。跟随软链接可能把根目录之外的文件搬进来；用
  `os.path.islink` + `follow_symlinks=False` 双重拦掉，属于安全边界。

* **移动前后都做边界校验**。目标路径必须落在分类根目录之内
  （`resolve()` 之后再比较），杜绝 `..` 与软链接把文件搬到根目录外面去。
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import ClassifierSettings, get_settings
from app.services.repo.language import LANGUAGE_LABELS, detect_language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 语言与归档目录的对应关系
# ---------------------------------------------------------------------------
# 需要新增语言支持时只改这里：例如加上 "go": "go"，扫描就会多出一个 go/ 归档目录。
ARCHIVE_DIRS: dict[str, str] = {
    "c": "c",
    "java": "java",
    "python": "python",
}

# 未知类型文件的标识与归档目录名
UNKNOWN_LANGUAGE = "unknown"
UNKNOWN_DIR = "unknown"

# 本次支持归档的语言（顺序固定，便于生成稳定的文档与测试）
ARCHIVABLE_LANGUAGES: tuple[str, ...] = tuple(ARCHIVE_DIRS)

# 根目录下这些名字的**目录**是"收纳盒"，扫描时要跳过，否则会自己搬自己
RESERVED_DIR_NAMES: frozenset[str] = frozenset({*ARCHIVE_DIRS.values(), UNKNOWN_DIR})

# 语言标识 -> 展示名。三种已知语言的名称复用全局表（保证全项目叫法一致），
# "未知"用需求里要求的字面写法（全局表里叫「未知语言」，这里统一成「未知」）
LANGUAGE_DISPLAY: dict[str, str] = {
    **{name: LANGUAGE_LABELS.get(name, name) for name in ARCHIVABLE_LANGUAGES},
    UNKNOWN_LANGUAGE: "未知",
}

# 重名时最多尝试多少个后缀（_1 ... _999），超了说明这个目录已经不正常了
MAX_RENAME_ATTEMPTS = 1000


@dataclass(slots=True)
class UploadResult:
    """把前端拖进来的文件保存到项目库的结果。"""

    filename: str            # 实际写入的文件名（重名会被改名）
    relative_path: str       # 相对分类根目录的路径，如 "homework/main.py"
    absolute_path: str
    size_bytes: int
    language: str            # c / java / python / unknown
    language_label: str
    saved: bool = True       # False 表示判定为重复，没有写入
    duplicate_of: str = ""   # 判定重复时，给出已存在文件的位置
    renamed: bool = False    # 是否因为重名而改了文件名
    note: str = ""


class FileClassifierError(RuntimeError):
    """分类过程中的可预期错误（调用方应转成 4xx，而不是 500）。"""


class FileClassifierPathError(FileClassifierError):
    """目录参数非法或越界。"""


class DuplicateFileError(FileClassifierError):
    """文件已存在（同名且内容相同）。

    单独定义一个异常类型，是为了让接口层能把它映射成 **409 Conflict** 而不是 400——
    前端据此提示"文件已存在"，与"参数不合法"区分开。
    """

    def __init__(self, message: str, *, existing_path: str = "") -> None:
        super().__init__(message)
        self.existing_path = existing_path


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ClassifiedFile:
    """一个被扫描到的文件及其处理结果。"""

    filename: str                 # 文件名（含扩展名）
    language: str                 # c / java / python / unknown
    language_label: str           # C / Java / Python / 未知（展示用）
    source_path: str              # 扫描时的相对路径（正斜杠），如 "homework/main.c"
    target_path: str              # 处理后的相对路径；未移动时与 source_path 相同
    absolute_path: str            # 处理后的绝对路径（写库用）
    size_bytes: int               # 文件大小（字节）
    modified_at: datetime | None  # 文件自身的修改时间
    action: str                   # moved / planned / kept / skipped
    note: str = ""                # 说明：为什么这么处理


@dataclass(slots=True)
class ClassificationReport:
    """一次分类扫描的完整结果。"""

    root: str                                   # 分类根目录（归档目录所在处）的绝对路径
    scanned_root: str = ""                      # 本次实际遍历的目录（可能是根目录的子目录）
    files: list[ClassifiedFile] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)   # 语言 -> 文件数
    truncated: bool = False                     # 是否触及 max_files 上限
    dry_run: bool = False                       # 是否为"只看不搬"的预演
    duration_ms: float = 0.0
    scanned_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # -- 便于阅读的统计 --------------------------------------------------
    @property
    def total(self) -> int:
        """参与本次分类的文件总数。"""
        return len(self.files)

    @property
    def moved(self) -> int:
        """实际（或预演中计划）归档的文件数。"""
        return sum(1 for item in self.files if item.action in {"moved", "planned"})

    @property
    def unknown(self) -> int:
        """识别为未知类型的文件数。"""
        return sum(1 for item in self.files if item.language == UNKNOWN_LANGUAGE)

    @property
    def skipped(self) -> int:
        """因软链接/超大/读不到而跳过的文件数。"""
        return sum(1 for item in self.files if item.action == "skipped")


# ---------------------------------------------------------------------------
# 主服务
# ---------------------------------------------------------------------------
class FileClassifier:
    """本地代码文件分类器：无状态，配置在构造时注入。"""

    def __init__(self, settings: ClassifierSettings | None = None) -> None:
        # 允许直接注入配置（测试用），默认取全局配置里的 classifier 段
        self.settings = settings or get_settings().classifier

    # ------------------------------------------------------------------
    # 目录解析（安全边界）
    # ------------------------------------------------------------------
    def resolve_root(self, directory: str | None = None) -> Path:
        """把接口传入的目录参数，解析成一个"确实位于分类根目录之内"的绝对路径。

        参数:
            directory: 相对分类根目录的子目录，例如 `"homework"`；
                       留空或传 `None` 表示根目录本身。
                       也允许直接传分类根目录的绝对路径（方便命令行/脚本调用）。

        返回:
            已 `resolve()` 过的绝对路径。

        异常:
            FileClassifierPathError —— 目录不存在、不是目录，或越出了根目录。

        关键逻辑:
            只允许"配置好的根目录及其子目录"。判断前先 `resolve()` 展开软链接，
            再做包含关系比较，因此 `../..`、指向外部的软链接都会被拦住。
            想扫描别的盘符，请改后端配置 `CLASSIFIER__ROOT`，
            **而不是**从接口传路径进来 —— 这个接口一旦能被指定任意路径，
            就等于给本机文件系统开了个搬运工。
        """
        root = self.settings.root_path
        try:
            root_real = root.resolve(strict=True)
        except OSError as exc:
            raise FileClassifierPathError(
                f"分类根目录不存在：{root}（请先在 .env 里配置 CLASSIFIER__ROOT，或创建该目录）"
            ) from exc

        if not root_real.is_dir():
            raise FileClassifierPathError(f"分类根目录不是目录：{root_real}")

        if directory is None or not str(directory).strip():
            return root_real

        candidate = Path(str(directory).strip())
        # 绝对路径也接受，但必须就是根目录本身或它的子目录
        target = candidate if candidate.is_absolute() else root_real / candidate
        try:
            target_real = target.resolve(strict=True)
        except OSError as exc:
            raise FileClassifierPathError(f"目录不存在：{directory}") from exc

        if not target_real.is_dir():
            raise FileClassifierPathError(f"不是目录：{directory}")
        if target_real != root_real and not target_real.is_relative_to(root_real):
            raise FileClassifierPathError(
                f"只能扫描 {root_real} 及其子目录，不接受根目录之外的路径：{directory}"
            )
        return target_real

    # ------------------------------------------------------------------
    # 语言识别
    # ------------------------------------------------------------------
    @staticmethod
    def detect_language(filename: str) -> str:
        """按扩展名判断语言，只返回 c / java / python / unknown 四种之一。

        参数:
            filename: 文件名或任意路径，只看扩展名，大小写不敏感。

        返回:
            `"c"` / `"java"` / `"python"`；不在支持范围内的返回 `"unknown"`。

        关键逻辑:
            先复用全局的扩展名表（`.c`、`.h` → c，`.java` → java，`.py` / `.pyi` → python），
            再判断它是否属于本功能支持归档的三种语言。这样 `.go` 会被识别出
            "这是 Go"，但按需求记成未知 —— 见 `_unknown_note()` 里给出的说明文字，
            避免学生以为自己的 .go 文件坏了。
        """
        detected = detect_language(filename)
        if detected in ARCHIVE_DIRS:
            return detected
        return UNKNOWN_LANGUAGE

    @staticmethod
    def _unknown_note(filename: str) -> str:
        """给"未知类型"的文件补一句人话说明，而不是只留一个 unknown。

        这里查的是**全局语言表**（含 Go / JavaScript 等），而不是只含四种取值的
        `LANGUAGE_DISPLAY`：`.go` 要说成「扩展名对应 Go」，不能写成「对应 go」。
        """
        detected = detect_language(filename)
        if detected is None:
            return "扩展名不在支持范围内（本功能只归档 .c/.h/.java/.py）"
        label = LANGUAGE_LABELS.get(detected, detected)
        return f"扩展名对应 {label}，不在本次支持的三种语言内，按未知处理"

    # ------------------------------------------------------------------
    # 前端拖拽入库：把文件保存到分类根目录
    # ------------------------------------------------------------------
    def save_upload(self, filename: str, content: bytes) -> UploadResult:
        """把前端拖进来的文件保存到项目库根目录，等 `/files/scan` 按语言归档。

        参数:
            filename: 浏览器给的文件名。**只取其中的文件名部分**，
                      任何路径成分都会被丢掉（防止 `../../evil.py` 这种写法越界）。
            content:  文件字节内容。

        返回:
            `UploadResult`。

        异常:
            FileClassifierPathError: 文件名为空/非法，或后缀不在支持范围内。
            FileClassifierError:     文件为空，或超过单文件大小上限。
            DuplicateFileError:      同名**且内容相同**的文件已经在项目库里。

        关键逻辑（"拖进项目库"这个动作的全部规则）:
            1. **只认三种语言的扩展名**：项目库是给 C / Java / Python 作业用的；
               拖进来一个 .zip 或 .exe 时立刻拒绝，比默默存进去更有用。
            2. **重复判定用「同名 + 内容相同」两个条件**：
               同一个文件重复拖入 → 报"文件已存在"（前端提示，不重复添加）；
               但**同名不同内容**（两个学生都交了 main.c）不能拒绝——
               那会让学生以为自己的作业没交上，所以自动改名 `main_1.c` 并说明。
            3. **只写进根目录，不直接写归档目录**：归档由 `/files/scan` 统一完成，
               这样"归档规则"永远只有一处实现，不会出现两条路径规则不一致。
            4. 查找重名时会连归档目录一起找（根目录 + c/ java/ python/ unknown/），
               否则第一次归档后再拖同一个文件就查不出重复了。
        """
        # ---- 1) 文件名清洗：只保留文件名本体 ----
        clean_name = Path(str(filename or "").replace("\\", "/")).name.strip()
        if not clean_name or clean_name in {".", ".."}:
            raise FileClassifierPathError("文件名不合法，无法入库")

        language = self.detect_language(clean_name)
        if language not in ARCHIVE_DIRS:
            raise FileClassifierPathError(
                f"只支持 {('、'.join(ARCHIVE_DIRS.values()))} 文件（.c/.h/.java/.py），"
                f"{clean_name} 不在支持范围内"
            )

        # ---- 2) 内容检查 ----
        if not content:
            raise FileClassifierError("文件是空的，没有内容可入库")
        if len(content) > self.settings.max_file_bytes:
            raise FileClassifierError(
                f"文件 {len(content) // 1024} KB 超过上限 "
                f"{self.settings.max_file_bytes // 1024} KB"
            )

        root = self.settings.root_path
        root.mkdir(parents=True, exist_ok=True)

        # ---- 3) 重复判定：同名 + 内容相同 ----
        digest = hashlib.sha256(content).hexdigest()
        existing = self._find_by_name(root, clean_name)
        for existed in existing:
            try:
                if hashlib.sha256(existed.read_bytes()).hexdigest() == digest:
                    relative = self._relative(root, existed)
                    raise DuplicateFileError(
                        f"文件已存在：{relative}", existing_path=relative
                    )
            except OSError:
                # 读不到就算了，继续找下一个同名文件；不能因为一个坏文件拒绝入库
                continue

        # ---- 4) 决定写盘用的文件名（同名但内容不同 → 改名，绝不覆盖）----
        #
        # 注意要**整个项目库**范围内判重名，不能只看根目录：
        # 第一次上传的文件经过 /files/scan 后已经搬到 python/ 之类的归档目录，
        # 此时根目录看起来"没有同名文件"，直接写进去就会出现两个同名文件
        # （一个在根、一个在 python/），前端列表里看着像重复添加，
        # 而且下一次归档又会撞名再改一次名。所以在写入前就把名字定好。
        target = self._unique_name_in_library(root, clean_name) if existing else root / clean_name
        renamed = target.name != clean_name
        try:
            target.write_bytes(content)
        except OSError as exc:
            raise FileClassifierError(f"保存文件失败：{exc}") from exc

        relative = self._relative(root, target)
        note = "已加入项目库，点「↻」或调用 /files/scan 即可按语言归档"
        if renamed:
            note = f"项目库里已有同名文件（内容不同），已另存为 {target.name}"

        return UploadResult(
            filename=target.name,
            relative_path=relative,
            absolute_path=str(target),
            size_bytes=len(content),
            language=language,
            language_label=LANGUAGE_DISPLAY.get(language, language),
            renamed=renamed,
            note=note,
        )

    def _unique_name_in_library(self, root: Path, filename: str) -> Path:
        """在**整个项目库**范围内取一个不冲突的文件名。

        参数:
            root:     分类根目录。
            filename: 期望的文件名。

        返回:
            可安全使用的路径（一定落在根目录下）。

        关键逻辑:
            与 `_unique_target` 的区别：后者只看"目标目录里有没有同名文件"，
            而这里要连归档目录（c/ java/ python/ unknown/）一起看。
            因为上传的文件先落在根目录、随后才被归档，
            只比较根目录会漏掉"归档目录里已经有同名文件"的情况。
        """
        stem, suffix = Path(filename).stem, Path(filename).suffix
        for index in range(0, MAX_RENAME_ATTEMPTS):
            candidate_name = filename if index == 0 else f"{stem}_{index}{suffix}"
            if not self._find_by_name(root, candidate_name):
                return root / candidate_name
        raise FileClassifierError(f"项目库里同名文件过多，放弃处理：{filename}")

    def _find_by_name(self, root: Path, filename: str) -> list[Path]:
        """在项目库里找出所有同名文件（根目录 + 各归档目录）。

        参数:
            root:     分类根目录。
            filename: 文件名。

        返回:
            磁盘上所有同名文件的路径列表。

        关键逻辑:
            必须连归档目录一起找：文件第一次归档后就从根目录搬到了 `c/` 之类的
            子目录，只看根目录的话，"重复拖入"永远查不出来。
            只查固定几个归档目录（不递归全树），因此开销与文件数无关。
        """
        candidates: list[Path] = [root / filename]
        for dirname in RESERVED_DIR_NAMES:
            candidates.append(root / dirname / filename)
        return [path for path in candidates if path.is_file()]

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------
    def classify(
        self,
        *,
        directory: str | None = None,
        dry_run: bool = False,
        move_unknown: bool | None = None,
        recursive: bool | None = None,
        max_files: int | None = None,
    ) -> ClassificationReport:
        """扫描目录、识别语言，并把文件归档到对应语言目录。

        参数:
            directory:    要扫描的子目录（相对根目录），留空表示整个根目录。
            dry_run:      True = 只预演不落地：不移动文件，也不写数据库，仅返回"打算怎么搬"。
                          第一次用、或者不确定目录里有什么时，建议先跑一次预演。
            move_unknown: 未知文件是否也移到 `unknown/`。不传则用配置里的默认值
                          （默认 False：只登记、不移动）。
            recursive:    是否递归子目录。不传则用配置默认值。
            max_files:    本次最多处理多少个文件，不传则用配置默认值。

        返回:
            `ClassificationReport`，包含每个文件的处理结果与按语言的统计。

        异常:
            FileClassifierPathError —— 目录参数非法/越界。

        关键逻辑（执行顺序很重要）:
            1. 先解析并校验目录，把边界确定下来；
            2. 遍历收集候选文件（跳过归档目录、排除目录、软链接、超大文件）；
            3. 逐个决定目标路径（重名就加后缀），**先算完再动手**，
               这样 dry_run 和真实执行走的是同一套决策代码，预演结果可信；
            4. 真实执行时创建目录并移动文件，全程只做"移动到根目录之内"这一种操作，
               不删除、不覆盖、不重命名已有文件。

        注意 `directory` 与归档位置的区分（很容易写错的地方）：
            `directory` 只决定**从哪里找文件**；归档目录永远是
            `c/` `java/` `python/` 挂在**配置的分类根目录**下，
            不会跟着 `directory` 跑到子目录里去建。
            也就是说 `directory="homework"` 时，`homework/sort.py` 依然会被
            移到 `<根目录>/python/sort.py`，而不是 `<根目录>/homework/python/`。
        """
        started = time.perf_counter()
        # scan_root = 去哪里找文件；archive_root = 归档目录挂在哪（恒为配置根目录）
        scan_root = self.resolve_root(directory)
        archive_root = self.settings.root_path.resolve()

        use_recursive = self.settings.recursive if recursive is None else recursive
        use_unknown_move = (
            self.settings.move_unknown if move_unknown is None else move_unknown
        )
        limit = self.settings.max_files if max_files is None else max_files

        report = ClassificationReport(
            root=str(archive_root), scanned_root=str(scan_root), dry_run=dry_run
        )

        for path in self._iter_candidates(scan_root, recursive=use_recursive):
            if len(report.files) >= limit:
                report.truncated = True
                logger.warning(
                    "分类扫描达到上限 %s 个文件，已提前停止（目录=%s）", limit, scan_root
                )
                break
            report.files.append(
                self._handle(
                    path,
                    archive_root=archive_root,
                    dry_run=dry_run,
                    move_unknown=use_unknown_move,
                )
            )

        report.counts = dict(Counter(item.language for item in report.files).most_common())
        report.duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "文件分类完成: 扫描目录=%s 文件=%s 统计=%s 归档=%s 未知=%s 预演=%s",
            scan_root, report.total, report.counts, report.moved, report.unknown, dry_run,
        )
        return report

    # ------------------------------------------------------------------
    # 遍历
    # ------------------------------------------------------------------
    def _iter_candidates(self, root: Path, *, recursive: bool):
        """遍历目录，产出"可能是学生代码"的文件路径（生成器）。

        参数:
            root:      已校验的绝对路径。
            recursive: True = 递归子目录；False = 只看 root 这一层。

        产出:
            磁盘上的文件路径（跳过项在这里就被过滤掉，不进入后续流程）。

        关键逻辑:
            用 `os.scandir` 手动遍历而不是 `rglob`，原因有两个：
              1. 需要在**根目录这一层**跳过归档目录（`c/` `java/` `python/` `unknown/`），
                 否则第二次扫描时会把已经归好的文件再搬一次；
                 注意只有根目录下的同名目录才跳过，嵌套的 `src/c/` 是正常代码目录；
              2. 需要在"不跟随软链接"的前提下判断是不是目录，
                 `entry.is_dir(follow_symlinks=False)` 正好能做到。
        """
        stack: list[Path] = [root]
        while stack:
            current = stack.pop()
            try:
                entries = sorted(os.scandir(current), key=lambda e: e.name.lower())
            except OSError as exc:  # 某个子目录没权限，不该让整次扫描失败
                logger.debug("跳过无法读取的目录 %s: %s", current, exc)
                continue

            for entry in entries:
                path = Path(entry.path)

                # 软链接一律跳过：可能指向根目录之外，搬进来就越界了
                if entry.is_symlink():
                    continue

                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue

                if is_dir:
                    if not recursive:
                        continue
                    # 只跳过"根目录下的"归档目录；嵌套同名目录照常扫描
                    if path.parent == root and entry.name in RESERVED_DIR_NAMES:
                        continue
                    if entry.name in self.settings.excluded_dirs:
                        continue
                    stack.append(path)
                    continue

                if not entry.is_file(follow_symlinks=False):
                    continue
                yield path

    # ------------------------------------------------------------------
    # 单个文件
    # ------------------------------------------------------------------
    def _handle(
        self, path: Path, *, archive_root: Path, dry_run: bool, move_unknown: bool
    ) -> ClassifiedFile:
        """处理一个文件：识别语言 → 决定目标位置 → （按需）移动到目标位置。

        参数:
            path:         文件在磁盘上的绝对路径。
            archive_root: 分类根目录（已 resolve），归档目录挂在它下面，
                          同时用于算相对路径与做边界检查。
            dry_run:      True 时只返回"打算怎么做"，不真的移动。
            move_unknown: 未知文件是否也归档到 `unknown/`。

        返回:
            `ClassifiedFile`，`action` 取值含义：
              * `moved`   已移动到语言目录；
              * `planned` 预演模式下的"将会移动"；
              * `kept`    保留原位（未知文件且未开启 move_unknown）；
              * `skipped` 未处理（软链接/超大/读不到），此时只有文件名与原因有效。
        """
        filename = path.name
        language = self.detect_language(filename)
        # 相对路径一律相对**分类根目录**算：即使本次只扫了某个子目录，
        # 数据库里也能还原出文件的完整位置（`homework/sort.py`）。
        source_rel = self._relative(archive_root, path)

        # ---- 取文件元信息；取不到就跳过（文件可能刚好被删/被占用）----
        try:
            stat = path.stat()
        except OSError as exc:
            logger.debug("无法读取文件信息，跳过 %s: %s", path, exc)
            return ClassifiedFile(
                filename=filename,
                language=language,
                language_label=LANGUAGE_DISPLAY.get(language, language),
                source_path=source_rel,
                target_path=source_rel,
                absolute_path=str(path),
                size_bytes=0,
                modified_at=None,
                action="skipped",
                note=f"读取文件信息失败：{exc}",
            )

        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)

        # ---- 大小上限：超大文件基本不是手写作业（可能是日志、数据集），只登记不搬 ----
        if stat.st_size > self.settings.max_file_bytes:
            return ClassifiedFile(
                filename=filename,
                language=language,
                language_label=LANGUAGE_DISPLAY.get(language, language),
                source_path=source_rel,
                target_path=source_rel,
                absolute_path=str(path),
                size_bytes=stat.st_size,
                modified_at=modified_at,
                action="skipped",
                note=(
                    f"文件 {stat.st_size // 1024} KB 超过上限 "
                    f"{self.settings.max_file_bytes // 1024} KB，未移动"
                ),
            )

        # ---- 决定要不要搬、搬去哪里 ----
        should_move = language in ARCHIVE_DIRS or (move_unknown and language == UNKNOWN_LANGUAGE)
        if not should_move:
            note = self._unknown_note(filename) if language == UNKNOWN_LANGUAGE else ""
            return ClassifiedFile(
                filename=filename,
                language=language,
                language_label=LANGUAGE_DISPLAY.get(language, language),
                source_path=source_rel,
                target_path=source_rel,
                absolute_path=str(path),
                size_bytes=stat.st_size,
                modified_at=modified_at,
                action="kept",
                note=note or "类型未知，按配置保留在原位置（只登记不移动）",
            )

        target_dir = archive_root / (ARCHIVE_DIRS.get(language) or UNKNOWN_DIR)
        target = self._unique_target(target_dir / filename)
        target_rel = self._relative(archive_root, target)
        self._assert_inside(archive_root, target)

        if dry_run:
            return ClassifiedFile(
                filename=filename,
                language=language,
                language_label=LANGUAGE_DISPLAY.get(language, language),
                source_path=source_rel,
                target_path=target_rel,
                absolute_path=str(target),
                size_bytes=stat.st_size,
                modified_at=modified_at,
                action="planned",
                note="预演：未实际移动",
            )

        note = "已归档"
        if target.name != filename:
            # 重名被改名了，必须明确告诉学生，否则他会以为文件丢了
            note = f"已归档（目标目录已有同名文件，重命名为 {target.name}）"

        self._move(path, target)
        return ClassifiedFile(
            filename=filename,
            language=language,
            language_label=LANGUAGE_DISPLAY.get(language, language),
            source_path=source_rel,
            target_path=target_rel,
            absolute_path=str(target),
            size_bytes=stat.st_size,
            modified_at=modified_at,
            action="moved",
            note=note,
        )

    # ------------------------------------------------------------------
    # 路径工具
    # ------------------------------------------------------------------
    @staticmethod
    def _relative(root: Path, path: Path) -> str:
        """算出相对根目录的路径，统一用正斜杠（跨平台、也方便前端展示）。"""
        try:
            return path.relative_to(root).as_posix()
        except ValueError:  # 理论上不会发生，兜底返回文件名
            return path.name

    @staticmethod
    def _assert_inside(root: Path, target: Path) -> None:
        """确认目标路径确实落在根目录之内，否则抛错。

        参数:
            root:   分类根目录（已 resolve）。
            target: 打算写入/移动到的路径。

        异常:
            FileClassifierPathError —— 目标越界，拒绝执行。

        关键逻辑:
            比较前先 `resolve()`：这样 `..` 与软链接都会被展开成真实路径，
            光看字符串前缀是拦不住 `root/link/../../etc` 这类写法的。
        """
        try:
            target_real = target.resolve()
        except OSError:  # 目标还不存在时 resolve 也可能失败，退化为父目录判断
            target_real = target.parent.resolve() / target.name
        if target_real != root and not target_real.is_relative_to(root):
            raise FileClassifierPathError(
                f"拒绝把文件移动到分类根目录之外：{target_real}"
            )

    @staticmethod
    def _unique_target(target: Path) -> Path:
        """目标已存在时，生成一个不冲突的新名字。

        参数:
            target: 期望的目标路径，如 `/codes/c/main.c`。

        返回:
            可安全使用的路径；已存在时依次尝试 `main_1.c`、`main_2.c`……

        异常:
            FileClassifierError —— 试满 `MAX_RENAME_ATTEMPTS` 次仍然冲突。

        关键逻辑:
            绝不覆盖。两个子文件夹里各有一个 `main.c` 是极常见的情况，
            直接 `shutil.move` 会把先搬过去的那份悄悄覆盖掉 —— 那就是丢作业。
        """
        if not target.exists():
            return target
        stem, suffix = target.stem, target.suffix
        for index in range(1, MAX_RENAME_ATTEMPTS + 1):
            candidate = target.with_name(f"{stem}_{index}{suffix}")
            if not candidate.exists():
                return candidate
        raise FileClassifierError(f"目标目录重名文件过多，放弃处理：{target}")

    @staticmethod
    def _move(source: Path, target: Path) -> None:
        """把文件移动到目标位置（目标目录会自动创建）。

        参数:
            source: 源文件绝对路径。
            target: 目标绝对路径（调用方需保证它不与已有文件冲突）。

        异常:
            FileClassifierError —— 移动失败（权限、被占用、磁盘满等）。

        关键逻辑:
            用 `shutil.move` 而不是 `os.rename`：同学把文件放在 U 盘或不同盘符时，
            `os.rename` 会直接抛"跨设备链接无效"，而 `shutil.move` 会自动退化为
            "复制 + 删除"，表现一致。
        """
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
        except (OSError, shutil.Error) as exc:
            raise FileClassifierError(f"移动文件失败：{source} -> {target}（{exc}）") from exc


__all__ = [
    "ARCHIVABLE_LANGUAGES",
    "ARCHIVE_DIRS",
    "LANGUAGE_DISPLAY",
    "RESERVED_DIR_NAMES",
    "UNKNOWN_DIR",
    "UNKNOWN_LANGUAGE",
    "ClassificationReport",
    "ClassifiedFile",
    "DuplicateFileError",
    "FileClassifier",
    "FileClassifierError",
    "FileClassifierPathError",
    "UploadResult",
]
