"""本地项目库「写入 / 删除」测试：在线编辑保存、原子落盘、回收站与只读开关。

为什么这些用例要单独守着一个文件：读取接口最坏的后果是「看到了不该看的文件」，
而写入接口最坏的后果是**把学生辛苦写的作业改坏或删掉**，而且往往不可恢复。
所以这里逐条钉住四件事：

1. **写进去的字节就是我要的字节**，包括换行符习惯：CRLF 文件保存后必须还是
   CRLF。统一改成 LF 的话，学生用记事本打开会看到「整个文件都被改过了」，
   连 AI 改错的结果都没法逐行核对。
2. **任何一步失败都必须原地不动**。保存走的是「同目录临时文件 + os.replace」，
   所以在乐观锁冲突 / 内容是二进制 / 超过大小上限 / 路径越界这几种情况下，
   原文件一个字节都不能变，也不能留下 `.dsh-tmp` 垃圾。
3. **删除默认只是移进 `.trash/`**（还能找回），同名条目不能互相覆盖，
   而 `.trash` 里的东西也不能再回到扫描列表里——否则「删了又冒出来」。
4. **接口层的状态码要准确**：400 = 请求不合法、403 = 只读模式、
   409 = 乐观锁冲突、413 = 太大。全都是 4xx 而不是 500，
   前端才能把 `detail` 里的中文提示原样显示给同学。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import datetime, tzinfo
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.services.library_service import (
    TEMP_SUFFIX,
    LibraryConflictError,
    LibraryFileTooLargeError,
    LibraryPathError,
    LibraryReadOnlyError,
    LibraryService,
    LibraryWriteError,
    _unique_path,
)

# 样例内容：一份最普通的 Python 作业（LF），一份 C 作业（手动拼 CRLF 用）
PY_SAMPLE = "def add(a, b):\n    return a + b\n"
NEW_PY_SAMPLE = "def add(a, b):\n    return a + b + 1\n"
NEW_C_SAMPLE = "int main(void) {\n    return 1;\n}\n"
CRLF_C_SAMPLE = "int main(void) {\r\n    return 0;\r\n}\r\n"


def _has_chinese(text: str) -> bool:
    """判断提示文本里有没有中文。

    报错信息是**给学生看的**，必须是看得懂的中文（如「文件不存在或不在项目库
    范围内」），不能漏出英文异常名或堆栈——前端会把 detail 原样显示在页面上。
    """
    return any("\u4e00" <= char <= "\u9fff" for char in text)


class _FrozenDatetime(datetime):
    """把 ``datetime.now()`` 钉在固定的一秒，用来稳定复现「同一秒内删两次」。

    产品代码用 ``datetime.now().strftime("%Y%m%d-%H%M%S")`` 拼回收站文件名。
    如果只是"飞快地连删两次"，绝大多数情况确实落在同一秒，但每次跨过整秒
    就偶发失败——测试不能靠运气，所以这里直接把时间冻住。
    """

    @classmethod
    def now(cls, tz: tzinfo | None = None) -> _FrozenDatetime:
        return cls(2026, 1, 1, 1, 1, 1, tzinfo=tz)


# ---------------------------------------------------------------------------
# 夹具：造一个临时项目库，避免碰到开发者真实的 data/library
# ---------------------------------------------------------------------------
@pytest.fixture()
def library_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """把项目库根目录指向临时文件夹（每个用例一份，互不干扰）。

    注意一律用 ``write_bytes`` 而不是 ``write_text``：Windows 上 write_text 会把
    ``\\n`` 自动翻译成 ``\\r\\n``（文本模式换行转换），而本文件有一半用例就是靠
    「原始字节到底是什么」来断言换行符有没有被偷偷改掉，用文本写入等于自己先把
    夹具改坏，再拿被改坏的基准去断言。
    """
    root = tmp_path / "mycode"
    (root / "homework").mkdir(parents=True)
    (root / "homework" / "utils.py").write_bytes(PY_SAMPLE.encode("utf-8"))
    (root / "homework" / "crlf.c").write_bytes(CRLF_C_SAMPLE.encode("utf-8"))

    monkeypatch.setenv("LIBRARY__ROOTS", str(root))
    get_settings.cache_clear()
    yield root
    get_settings.cache_clear()


@pytest.fixture()
def service(library_root: Path) -> LibraryService:
    """按当前配置构造服务（库根目录已被 library_root 指到临时目录）。"""
    return LibraryService(get_settings().library)


@pytest.fixture()
def read_only_service(
    library_root: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[LibraryService]:
    """把项目库切成只读（``LIBRARY__ALLOW_WRITE=false``）后再构造服务。

    这个开关是给「项目库指向老师共享的公共目录」用的：同学能看、能送进 AI 分析，
    但不能改公共文件。注意服务是**按配置构造**的，所以改了配置必须重新构造一个，
    不能继续用 ``service`` 夹具里那个（它攥着旧的 settings）。
    """
    monkeypatch.setenv("LIBRARY__ALLOW_WRITE", "false")
    get_settings.cache_clear()
    yield LibraryService(get_settings().library)
    get_settings.cache_clear()


@pytest.fixture()
def library_app(sqlite_path: Path) -> Iterator[object]:
    """真实应用（项目库接口不调用大模型，无需注入假客户端）。"""
    app = create_app()
    yield app


# ---------------------------------------------------------------------------
# 服务层：覆盖保存
# ---------------------------------------------------------------------------
def test_write_text_overwrites_content(service: LibraryService, library_root: Path) -> None:
    """保存后磁盘上的字节必须真的变了，而且返回的元信息全部对得上新内容。

    这是「在线编辑保存」的地基。后端把写回后的 sha256 交给前端当下一轮的乐观锁
    基准，如果 size_bytes / line_count / sha256 里任何一个还是旧文件的，
    前端就会拿着错误的基准继续编辑，下一次保存必然 409——同学只会看到
    「保存按钮坏了」。所以这里既查返回值，也重新读一遍文件核对。
    """
    target = library_root / "homework" / "utils.py"
    old_bytes = target.read_bytes()

    result = service.write_text("homework/utils.py", NEW_PY_SAMPLE)
    new_bytes = target.read_bytes()

    assert new_bytes != old_bytes
    assert new_bytes == NEW_PY_SAMPLE.encode("utf-8")
    assert result.rel_path == "homework/utils.py"
    assert result.filename == "utils.py"
    assert result.language == "python"
    assert result.size_bytes == len(new_bytes)
    assert result.char_count == len(NEW_PY_SAMPLE)
    assert result.line_count == 2
    # 指纹必须是**写入后**的内容指纹，而不是读进来时那个
    assert result.sha256 == hashlib.sha256(new_bytes).hexdigest()
    assert result.sha256 != hashlib.sha256(old_bytes).hexdigest()
    # 原子写入不能留下临时文件垃圾（否则下次扫描会看到一堆 .dsh-tmp）
    assert list((library_root / "homework").glob(f"*{TEMP_SUFFIX}")) == []

    # 再走一遍读接口：两个接口看到的必须是同一份磁盘状态
    content = service.read_file("homework/utils.py")
    assert content.code == NEW_PY_SAMPLE
    assert content.sha256 == result.sha256
    assert content.size_bytes == result.size_bytes
    assert content.line_count == result.line_count


def test_write_text_keeps_newline_convention(service: LibraryService, library_root: Path) -> None:
    """CRLF 文件保存后还是 CRLF，LF 文件保存后还是 LF（直接断言原始字节）。

    学生机上的作业几乎都是 CRLF。如果保存时图省事统一写成 LF，他用记事本打开
    会发现「每行都变了」、git diff 全红，AI 改错给出的结果也没法和原文对照。
    换行习惯必须跟着**原文件**走——这也正是响应里 newline 字段存在的理由。
    """
    crlf_target = library_root / "homework" / "crlf.c"
    crlf_result = service.write_text("homework/crlf.c", NEW_C_SAMPLE)  # 传进去的是 LF 内容

    assert crlf_result.newline == "\r\n"
    crlf_bytes = crlf_target.read_bytes()
    assert crlf_bytes == NEW_C_SAMPLE.replace("\n", "\r\n").encode("utf-8")
    # 不能有"漏网的裸 \n"：把 \r\n 全部摘掉之后不该再剩下换行
    assert crlf_bytes.replace(b"\r\n", b"").count(b"\n") == 0
    assert crlf_result.line_count == 3

    lf_target = library_root / "homework" / "utils.py"
    lf_result = service.write_text("homework/utils.py", NEW_C_SAMPLE.replace("\n", "\r\n"))

    assert lf_result.newline == "\n"
    lf_bytes = lf_target.read_bytes()
    assert b"\r" not in lf_bytes
    assert lf_bytes == NEW_C_SAMPLE.encode("utf-8")


def test_write_text_stale_sha_leaves_file_untouched(
    service: LibraryService, library_root: Path
) -> None:
    """乐观锁对不上时必须报冲突，而且**一个字节都不写**。

    场景很具体：同学在网页上打开了 a.py，又顺手用记事本改了同一个文件并保存。
    这时网页提交上来的 sha256 已经过期，如果照样覆盖，他在记事本里的改动就凭空
    消失了——所以宁可让他重新打开一次，也不能默默盖掉别人的（或自己的）新内容。
    """
    target = library_root / "homework" / "utils.py"
    before = target.read_bytes()

    with pytest.raises(LibraryConflictError):
        service.write_text("homework/utils.py", NEW_PY_SAMPLE, expected_sha256="0" * 64)

    assert target.read_bytes() == before
    assert not target.with_name(target.name + TEMP_SUFFIX).exists()


def test_write_text_accepts_matching_sha(service: LibraryService, library_root: Path) -> None:
    """指纹正确时保存必须成功——乐观锁不能变成「永远存不上」。

    这里的 sha256 就是前端打开文件时拿到的那个值，等价于「中途没人动过文件」
    的正常路径：一旦这条路径也被拦下，整个编辑功能就废了。
    """
    current = service.read_file("homework/utils.py").sha256

    result = service.write_text("homework/utils.py", NEW_PY_SAMPLE, expected_sha256=current)

    assert result.sha256 != current
    assert (library_root / "homework" / "utils.py").read_bytes() == NEW_PY_SAMPLE.encode("utf-8")


def test_write_text_rejects_nul_bytes(service: LibraryService, library_root: Path) -> None:
    """含 NUL 的内容一律拒绝：那是二进制（压缩包 / 图片），不是代码。

    同学用「用本机文件替换」时很容易误选一个 zip。真写进去这个文件就再也打不开
    了，连"改回来"都做不到——所以必须在落盘前拦下，并保持原文件原样。
    """
    target = library_root / "homework" / "utils.py"
    before = target.read_bytes()

    with pytest.raises(LibraryWriteError):
        service.write_text("homework/utils.py", "x = 1\x00\n")

    assert target.read_bytes() == before
    assert not target.with_name(target.name + TEMP_SUFFIX).exists()


def test_write_text_rejects_oversized_content(
    library_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """新内容超过单文件上限时必须报错，并且拒绝发生在落盘**之前**。

    上限存在的意义是不让一次误操作把几十 MB 的东西灌进项目库。如果先写后报错，
    文件已经被写坏了，再返回 413 也没用——所以这里同时断言原文件字节没变。

    注意：改了 ``LIBRARY__MAX_FILE_BYTES`` 之后必须重新构造服务，
    因为 ``service`` 夹具里那个实例攥着旧配置（这也是**故意不用**该夹具的原因）。
    """
    target = library_root / "homework" / "utils.py"
    before = target.read_bytes()

    monkeypatch.setenv("LIBRARY__MAX_FILE_BYTES", "64")
    get_settings.cache_clear()
    limited = LibraryService(get_settings().library)

    with pytest.raises(LibraryFileTooLargeError):
        limited.write_text("homework/utils.py", "x = 1\n" * 100)

    assert target.read_bytes() == before
    assert not target.with_name(target.name + TEMP_SUFFIX).exists()
    get_settings.cache_clear()


def test_write_text_rejects_traversal(service: LibraryService, library_root: Path) -> None:
    """越界路径必须被拒，而且**根目录之外不能凭空多出文件**。

    写入比读取更危险：读越界最坏是「看到了不该看的」，写越界是「改坏了别人的文件」。
    这里特意把「没写出去」也断言上，因为"抛了异常但文件已经落地"是最糟的组合。
    """
    escaped = (library_root / "../../evil.py").resolve()

    with pytest.raises(LibraryPathError):
        service.write_text("../../evil.py", "EVIL = 1\n")

    assert not escaped.exists()


def test_write_text_refuses_to_create_new_file(
    service: LibraryService, library_root: Path
) -> None:
    """文件不存在时直接报错，绝不能顺手新建一个。

    新建文件必须走「拖进项目库」的入库流程（那边会做编码识别、语言归档、
    大小限制）。如果保存接口能凭一个路径就在磁盘上创建文件，等于把「往任意位置
    写文件」的能力交给了请求体，前面辛苦设的安全边界就白做了。
    """
    with pytest.raises(LibraryPathError):
        service.write_text("homework/brand_new.py", "x = 1\n")

    assert not (library_root / "homework" / "brand_new.py").exists()
    # 连临时文件都不能出现（glob 同时覆盖 brand_new.py.dsh-tmp）
    assert list((library_root / "homework").glob("brand_new*")) == []


# ---------------------------------------------------------------------------
# 服务层：删除
# ---------------------------------------------------------------------------
def test_delete_file_moves_into_trash(service: LibraryService, library_root: Path) -> None:
    """默认删除是「移进 .trash」，所以文件还能被找回来（给手滑的同学兜底）。

    三件事必须同时成立：原路径没了、`.trash` 里真有一份内容相同的副本、
    返回的 trash_path 指的就是它——前端要靠这个路径告诉同学「去哪儿找回」。
    """
    target = library_root / "homework" / "utils.py"
    before = target.read_bytes()

    result = service.delete_file("homework/utils.py")

    assert result.permanent is False
    assert result.rel_path == "homework/utils.py"
    assert result.size_bytes == len(before)
    assert not target.exists()

    trash_dir = library_root / ".trash"
    entries = list(trash_dir.glob("*__*.py"))
    assert len(entries) == 1
    assert entries[0].read_bytes() == before
    assert Path(result.trash_path).exists()
    assert Path(result.trash_path).resolve() == entries[0].resolve()
    assert Path(result.trash_dir).name == ".trash"


def test_delete_file_permanent_unlinks_without_trash(
    service: LibraryService, library_root: Path
) -> None:
    """permanent=true 才是真删：文件消失，而且不该白白建出一个 .trash 目录。

    只读浏览的目录里（或同学明确选了「彻底删除」时）多留一个空的回收站目录，
    会让"项目库被谁动过"这件事变得难以解释。
    """
    target = library_root / "homework" / "utils.py"

    result = service.delete_file("homework/utils.py", permanent=True)

    assert result.permanent is True
    assert not target.exists()
    assert result.trash_path == ""
    assert not (library_root / ".trash").exists()


def test_delete_file_twice_reports_missing(service: LibraryService, library_root: Path) -> None:
    """第二次删除必须明确报「文件不存在」，而不是静默成功。

    前端把 400 的 detail 直接显示给同学。如果第二次也回 200，她会以为文件还在、
    只是删除没生效，于是反复点——所以"已经删过了"也是一个需要说清楚的状态。
    """
    service.delete_file("homework/utils.py")

    with pytest.raises(LibraryPathError):
        service.delete_file("homework/utils.py")


def test_delete_two_same_named_files_keeps_both_trash_entries(
    service: LibraryService, library_root: Path
) -> None:
    """两个子目录里的同名 main.py 都要完整地留在回收站里。

    回收站文件名是「时间戳__把 / 换成 __ 的相对路径」，同名文件在同一秒内被删
    很容易被以为会撞名。这里断言的是最关键的结果：**两份内容都在**。
    被覆盖掉的那一份是永远拿不回来的，而「能找回」正是回收站存在的唯一理由。
    """
    for folder, marker in (("a", "AAA"), ("b", "BBB")):
        target = library_root / folder / "main.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        # 字面量默认按 UTF-8 编码，这里没有非 ASCII 字符，不必再写一遍 encoding
        target.write_bytes(f"MARK = '{marker}'\n".encode())

    service.delete_file("a/main.py")
    service.delete_file("b/main.py")

    assert not (library_root / "a" / "main.py").exists()
    assert not (library_root / "b" / "main.py").exists()
    entries = sorted((library_root / ".trash").glob("*.py"))
    assert len(entries) == 2
    assert sorted(entry.read_bytes() for entry in entries) == [
        b"MARK = 'AAA'\n",
        b"MARK = 'BBB'\n",
    ]


def test_delete_same_path_twice_does_not_overwrite_trash(
    service: LibraryService, library_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同一秒内删掉**同一个相对路径**两次时，回收站里必须留下两份。

    这才是 `_unique_path` 真正守着的那种撞名：a/main.py 被删掉之后同学又从别处
    拷回来一份、再删一次，两次算出的回收站文件名（时间戳 + 扁平化路径）**完全
    一样**，后一次会把前一次那份覆盖掉。用例把 ``datetime.now()`` 冻住，
    让「同一秒」确定性地成立，而不是碰运气。
    """
    monkeypatch.setattr("app.services.library_service.datetime", _FrozenDatetime)
    target = library_root / "a" / "main.py"
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_bytes(b"VERSION = 1\n")
    service.delete_file("a/main.py")
    target.write_bytes(b"VERSION = 2\n")
    service.delete_file("a/main.py")

    entries = sorted((library_root / ".trash").glob("*.py"))
    assert len(entries) == 2
    assert sorted(entry.read_bytes() for entry in entries) == [
        b"VERSION = 1\n",
        b"VERSION = 2\n",
    ]
    # 两个名字带着同一个时间戳前缀 —— 证明"撞名"真的发生了，
    # 靠的是 _unique_path 让开，而不是两次删在不同的秒里。
    assert len({entry.name.split("__")[0] for entry in entries}) == 1
    assert len({entry.name for entry in entries}) == 2


def test_unique_path_makes_room_for_existing_target(tmp_path: Path) -> None:
    """`_unique_path` 撞名时要让开：加 `_1`，而不是把已存在的文件顶掉。

    这是上面那个"回收站不覆盖"用例的最小单元版本：即使删除流程将来被重写，
    这条"绝不覆盖已有文件"的规矩也要一直有测试守着。
    """
    taken = tmp_path / "20260101-010101__main.py"
    taken.write_bytes(b"OLD")

    first = _unique_path(taken)
    assert first == tmp_path / "20260101-010101__main_1.py"
    first.write_bytes(b"NEW")

    second = _unique_path(taken)
    assert second == tmp_path / "20260101-010101__main_2.py"
    # 已有文件一个字节都不能被动过
    assert taken.read_bytes() == b"OLD"
    assert first.read_bytes() == b"NEW"
    # 目标不存在时原样返回，不要凭空加后缀
    assert _unique_path(tmp_path / "nobody.py") == tmp_path / "nobody.py"


# ---------------------------------------------------------------------------
# 服务层：只读开关与回收站可见性
# ---------------------------------------------------------------------------
def test_read_only_mode_blocks_write_and_delete(
    read_only_service: LibraryService, library_root: Path
) -> None:
    """只读模式下写和删都要被挡住，并且文件原封不动。

    任何一条写路径漏掉这个检查，老师共享的公共目录就会被改乱——而这类毛病在
    开发机上完全看不出来（那里的 allow_write 默认是 true）。
    """
    target = library_root / "homework" / "utils.py"
    before = target.read_bytes()

    with pytest.raises(LibraryReadOnlyError):
        read_only_service.write_text("homework/utils.py", NEW_PY_SAMPLE)
    with pytest.raises(LibraryReadOnlyError):
        read_only_service.delete_file("homework/utils.py")

    assert target.read_bytes() == before
    assert not (library_root / ".trash").exists()
    # 接口层能先匹配到 403 而不是笼统的 400，靠的就是这个继承关系
    assert issubclass(LibraryReadOnlyError, LibraryWriteError)


def test_scan_ignores_trash_contents(service: LibraryService, library_root: Path) -> None:
    """.trash 里的文件不能再出现在扫描列表里。

    否则同学「删掉」的文件会立刻换个名字回到列表上，看起来像删除失败；
    更糟的是他可能再删一次，越删越多。所以 `.trash` 必须在排除目录名单里。
    """
    trash_dir = library_root / ".trash"
    trash_dir.mkdir(parents=True, exist_ok=True)
    (trash_dir / "20260101-010101__homework__utils.py").write_bytes(PY_SAMPLE.encode("utf-8"))

    rel_paths = [item.rel_path for item in service.scan().files]

    assert rel_paths == ["homework/crlf.c", "homework/utils.py"]
    assert not any(".trash" in path for path in rel_paths)


# ---------------------------------------------------------------------------
# 接口层：PUT /api/v1/library/file
# ---------------------------------------------------------------------------
def test_put_file_endpoint_saves_and_reads_back(library_root: Path, library_app: object) -> None:
    """PUT 保存成功要回 200 与新指纹，紧接着的 GET 必须读到新内容。

    这是前端「改完点保存」的完整回合：响应里的 sha256 会成为下一轮的乐观锁基准，
    所以两个接口必须看到同一份磁盘状态，否则同学每次保存都会被自己上一次的
    保存结果判成"文件被别处改过"。
    """
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.put(
            "/api/v1/library/file",
            json={"path": "homework/utils.py", "code": NEW_PY_SAMPLE},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["rel_path"] == "homework/utils.py"
        assert body["filename"] == "utils.py"
        assert body["language"] == "python"
        assert body["size_bytes"] == len(NEW_PY_SAMPLE.encode("utf-8"))
        assert body["char_count"] == len(NEW_PY_SAMPLE)
        assert body["line_count"] == 2
        assert body["newline"] == "\n"
        assert body["encoding"] == "utf-8"
        assert body["sha256"]
        assert body["message"]
        assert body["trace_id"]

        read_back = client.get("/api/v1/library/file", params={"path": "homework/utils.py"})

    assert read_back.status_code == 200, read_back.text
    read_body = read_back.json()
    assert read_body["code"] == NEW_PY_SAMPLE
    assert read_body["sha256"] == body["sha256"]
    assert read_body["line_count"] == body["line_count"]


def test_put_endpoint_traversal_is_400(library_root: Path, library_app: object) -> None:
    """越界保存要回 400（请求不合法）而不是 500，且提示是中文。

    前端把 detail 原样显示给同学，所以它必须是一句人话；
    越界又是同学最可能踩到的坑（从资源管理器复制来一个带 ../ 的路径）。
    """
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.put(
            "/api/v1/library/file",
            json={"path": "../../evil.py", "code": "EVIL = 1\n"},
        )

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert ".." in detail
    assert _has_chinese(detail)
    assert not (library_root / "../../evil.py").resolve().exists()


def test_put_endpoint_missing_file_is_400(library_root: Path, library_app: object) -> None:
    """对不存在的文件保存要回 400（而不是 404/500），并且不会把它创建出来。

    新建文件必须走入库流程；接口层这条规矩靠服务层的 LibraryPathError 保证，
    这里从 HTTP 这一侧再确认一次状态码与"确实没落盘"。
    """
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.put(
            "/api/v1/library/file",
            json={"path": "homework/nope.py", "code": "x = 1\n"},
        )

    assert response.status_code == 400, response.text
    assert _has_chinese(response.json()["detail"])
    assert not (library_root / "homework" / "nope.py").exists()


def test_put_endpoint_stale_sha_is_409(library_root: Path, library_app: object) -> None:
    """乐观锁冲突要给 409 而不是 400：前端才能弹「文件已被别处改动，请重新打开」。

    409 与 400 的区分是有意义的——400 会让前端把提示当成"你填错了"，
    而真正该做的是重新加载这个文件。
    """
    target = library_root / "homework" / "utils.py"
    before = target.read_bytes()

    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.put(
            "/api/v1/library/file",
            json={
                "path": "homework/utils.py",
                "code": NEW_PY_SAMPLE,
                "expected_sha256": "0" * 64,
            },
        )

    assert response.status_code == 409, response.text
    assert _has_chinese(response.json()["detail"])
    assert target.read_bytes() == before


def test_put_endpoint_read_only_is_403(
    library_root: Path,
    read_only_service: LibraryService,
    library_app: object,
) -> None:
    """只读模式下保存要回 403；前端据此把「编辑 / 替换」按钮禁掉。

    （这里只借用 read_only_service 夹具的副作用：把 LIBRARY__ALLOW_WRITE 设成
    false 并清掉配置缓存；请求用的是应用自己按当前配置构造的服务。）
    """
    target = library_root / "homework" / "utils.py"
    before = target.read_bytes()

    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.put(
            "/api/v1/library/file",
            json={"path": "homework/utils.py", "code": NEW_PY_SAMPLE},
        )

    assert response.status_code == 403, response.text
    assert _has_chinese(response.json()["detail"])
    assert target.read_bytes() == before


# ---------------------------------------------------------------------------
# 接口层：DELETE /api/v1/library/file
# ---------------------------------------------------------------------------
def test_delete_endpoint_moves_file_to_trash(library_root: Path, library_app: object) -> None:
    """DELETE 默认要回 200 且 permanent=false，并把文件真的挪进回收站。

    前端靠 permanent / trash_dir 决定提示文案（「已移到回收站，可去 xxx 找回」），
    所以这几个字段一个都不能空。
    """
    target = library_root / "homework" / "utils.py"
    before = target.read_bytes()

    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.delete("/api/v1/library/file", params={"path": "homework/utils.py"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["permanent"] is False
    assert body["rel_path"] == "homework/utils.py"
    assert body["size_bytes"] == len(before)
    assert body["trash_path"]
    assert Path(body["trash_path"]).read_bytes() == before
    assert Path(body["trash_dir"]).name == ".trash"
    assert _has_chinese(body["message"])
    assert not target.exists()


def test_delete_endpoint_traversal_is_400(library_root: Path, library_app: object) -> None:
    """越界删除要回 400：绝不能因为一个 ../ 就把项目库外面的文件删了。

    这里特意**先在外面放一个文件**再尝试删它：只看状态码是不够的，
    必须证明那个文件真的还在（文件名与 PUT 那个用例区分开，
    免得两个用例在 pytest 的临时目录里互相影响）。
    """
    escaped = (library_root / "../../evil_delete.py").resolve()
    escaped.write_bytes(b"IMPORTANT = 1\n")

    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.delete("/api/v1/library/file", params={"path": "../../evil_delete.py"})

    assert response.status_code == 400, response.text
    assert _has_chinese(response.json()["detail"])
    # 关键断言：根目录之外那个文件必须还在
    assert escaped.read_bytes() == b"IMPORTANT = 1\n"


def test_delete_endpoint_missing_file_is_400(library_root: Path, library_app: object) -> None:
    """删一个不存在的文件要回 400（学生手滑点了两次就是这个提示）。"""
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.delete("/api/v1/library/file", params={"path": "homework/nope.py"})

    assert response.status_code == 400, response.text
    assert _has_chinese(response.json()["detail"])


def test_delete_endpoint_read_only_is_403(
    library_root: Path,
    read_only_service: LibraryService,
    library_app: object,
) -> None:
    """只读模式下删除要回 403，并且文件必须还在（这是最不能被误删的场景）。"""
    target = library_root / "homework" / "utils.py"

    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.delete("/api/v1/library/file", params={"path": "homework/utils.py"})

    assert response.status_code == 403, response.text
    assert _has_chinese(response.json()["detail"])
    assert target.exists()
    assert not (library_root / ".trash").exists()


# ---------------------------------------------------------------------------
# 接口层：前端靠这些字段决定「按钮画不画」
# ---------------------------------------------------------------------------
def test_scan_endpoint_exposes_write_flags(library_root: Path, library_app: object) -> None:
    """扫描结果要带上 allow_write 与 trash_dir_name。

    前端拿不到这两个字段就只能靠猜：默认当成"可写"，于是在只读项目库上照样画出
    编辑/删除按钮，同学一点就吃 403。让接口说清楚，页面才不会自己编默认值。
    """
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.get("/api/v1/library/scan")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["allow_write"] is True
    assert body["trash_dir_name"] == ".trash"


def test_scan_endpoint_reports_read_only(
    library_root: Path,
    read_only_service: LibraryService,
    library_app: object,
) -> None:
    """只读时 allow_write 必须是 false——这正是前端禁用按钮的依据。

    只看服务端拒绝写是不够的：页面还照常给按钮，同学点一次吃一个 403，
    体验上就是"功能坏了"。所以这个字段要单独守着。
    """
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.get("/api/v1/library/scan")

    assert response.status_code == 200, response.text
    assert response.json()["allow_write"] is False


def test_openapi_keeps_write_endpoints(library_app: object) -> None:
    """OpenAPI 里必须同时有 PUT 与 DELETE /api/v1/library/file。

    前端是按接口文档写死的。将来谁"重构"掉这条路由（比如改名成 /file/write），
    页面上点保存只会得到 405，而且很难猜是后端的锅——这条用例让它当场变红。
    """
    schema = library_app.openapi()  # type: ignore[attr-defined]
    operations = schema["paths"]["/api/v1/library/file"]

    assert {"get", "put", "delete"} <= set(operations)
