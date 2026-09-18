"""本地项目库测试：扫描分类、编码兜底、路径越界的拦截、API 契约。

这里最需要盯住的是**安全边界**：项目库会读学生电脑上的文件，
所以"能不能读到根目录之外的东西"必须有测试守着。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import LibrarySettings, Settings, get_settings
from app.main import create_app
from app.services.library_service import (
    LibraryFileTooLargeError,
    LibraryPathError,
    LibraryService,
    count_lines,
    detect_language_by_name,
    language_label,
)
from app.services.text_utils import count_text_lines

PY_SAMPLE = "def add(a, b):\n    return a + b\n"
C_SAMPLE = "#include <stdio.h>\n\nint main(void) {\n    return 0;\n}\n"
JAVA_SAMPLE = "public class A {\n    int f() { return 1; }\n}\n"


# ---------------------------------------------------------------------------
# 夹具：造一个临时项目库，避免扫到开发者真实的 data/library
# ---------------------------------------------------------------------------
@pytest.fixture()
def library_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """把项目库根目录指向临时文件夹（每个用例一份，互不干扰）。

    注意一律用 ``write_bytes`` 而不是 ``write_text``：Windows 上 write_text 会把
    ``\\n`` 自动翻译成 ``\\r\\n``（文本模式换行转换），写入内容就和我期望的
    样例字符串对不上了，断言会莫名其妙地失败。用字节写入才能精确控制文件内容。
    """
    root = tmp_path / "mycode"
    (root / "homework").mkdir(parents=True)
    (root / "homework" / "linked_list.c").write_bytes(C_SAMPLE.encode("utf-8"))
    (root / "homework" / "Calculator.java").write_bytes(JAVA_SAMPLE.encode("utf-8"))
    (root / "homework" / "utils.py").write_bytes(PY_SAMPLE.encode("utf-8"))
    # 非源码文件：必须被忽略
    (root / "README.md").write_bytes("# 我的作业\n".encode())
    (root / "data.txt").write_bytes(b"1,2,3\n")
    # 排除目录里的代码：不是学生手写的作业，不该出现在列表里
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.js").write_bytes(b"function dep() {}\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "junk.py").write_bytes(b"x = 1\n")

    monkeypatch.setenv("LIBRARY__ROOTS", str(root))
    get_settings.cache_clear()
    yield root
    get_settings.cache_clear()


@pytest.fixture()
def service(library_root: Path) -> LibraryService:
    return LibraryService(get_settings().library)


@pytest.fixture()
def library_app(sqlite_path: Path) -> Iterator[object]:
    """真实应用（项目库接口不需要大模型，无需注入假客户端）。"""
    app = create_app()
    yield app


# ---------------------------------------------------------------------------
# 语言判定
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("a.c", "c"),
        ("a.h", "c"),
        ("A.java", "java"),
        ("main.py", "python"),
        ("app.js", "javascript"),
        ("main.go", "go"),
    ],
)
def test_detect_language_by_name(filename: str, expected: str) -> None:
    assert detect_language_by_name(filename) == expected


@pytest.mark.parametrize("filename", ["README.md", "data.txt", "photo.png", "noext"])
def test_unknown_suffix_is_not_code(filename: str) -> None:
    assert detect_language_by_name(filename) is None


def test_language_label_falls_back_to_identifier() -> None:
    assert language_label("python") == "Python"
    # 没登记过的语言不应该是 KeyError，直接原样显示即可
    assert language_label("cobol") == "cobol"


# ---------------------------------------------------------------------------
# 行数统计
# ---------------------------------------------------------------------------
def test_count_lines_edge_cases(tmp_path: Path) -> None:
    empty = tmp_path / "empty.py"
    empty.write_text("", encoding="utf-8")
    assert count_lines(empty) == 0

    one = tmp_path / "one.py"
    one.write_text("x = 1", encoding="utf-8")  # 末尾没有换行
    assert count_lines(one) == 1

    three = tmp_path / "three.py"
    three.write_text("a\nb\nc\n", encoding="utf-8")
    assert count_lines(three) == 3

    chinese = tmp_path / "cn.py"
    chinese.write_text("第一行\n第二行\n", encoding="utf-8")
    assert count_lines(chinese) == 2


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", 0),
        ("a", 1),
        ("a\n", 1),
        ("a\nb", 2),
        ("a\nb\n", 2),
        ("\n", 1),
        ("a\r\nb\r\n", 2),
    ],
)
def test_count_text_lines_rule(text: str, expected: int) -> None:
    assert count_text_lines(text) == expected


@pytest.mark.parametrize("text", ["", "a", "a\n", "a\nb", "a\r\nb\r\n", "\n\n\n"])
def test_text_and_file_line_count_agree(tmp_path: Path, text: str) -> None:
    """两个实现（字符串版 / 流式读文件版）必须给出同样的行数。

    它们服务不同场景（分析结果 vs 扫描列表），一旦口径漂移，
    同一个文件在页面上就会显示两个行数，学生一定会发现。
    """
    path = tmp_path / "sample.py"
    path.write_bytes(text.encode("utf-8"))
    assert count_lines(path) == count_text_lines(text)


# ---------------------------------------------------------------------------
# 扫描与分类
# ---------------------------------------------------------------------------
def test_scan_only_picks_code_files(service: LibraryService) -> None:
    result = service.scan()
    names = sorted(item.rel_path for item in result.files)
    assert names == [
        "homework/Calculator.java",
        "homework/linked_list.c",
        "homework/utils.py",
    ]


def test_scan_counts_by_language(service: LibraryService) -> None:
    result = service.scan()
    assert result.total_files == 3
    assert result.language_counts == {"c": 1, "java": 1, "python": 1}
    # 行数累加要对得上：C 5 行 + Java 3 行 + Python 2 行
    assert result.total_lines == 10


def test_scan_ignores_excluded_dirs(service: LibraryService) -> None:
    """排除目录里的代码一个都不能出现。"""
    paths = [item.rel_path for item in service.scan().files]
    assert not any("node_modules" in path for path in paths)
    assert not any("__pycache__" in path for path in paths)


def test_scan_reports_root_metadata(service: LibraryService, library_root: Path) -> None:
    result = service.scan()
    assert len(result.roots) == 1
    root = result.roots[0]
    assert root.exists is True
    assert root.file_count == 3
    assert root.path == str(library_root)


def test_scan_missing_root_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """目录不存在时应该老实报告，而不是直接把接口打崩。"""
    monkeypatch.setenv("LIBRARY__ROOTS", str(tmp_path / "not_here"))
    get_settings.cache_clear()
    result = LibraryService(get_settings().library).scan()
    assert result.roots[0].exists is False
    assert result.roots[0].skipped_reason == "目录不存在"
    assert result.files == []
    get_settings.cache_clear()


def test_scan_filters_by_language(service: LibraryService) -> None:
    result = service.scan(language="python")
    assert [item.rel_path for item in result.files] == ["homework/utils.py"]


def test_scan_filters_by_keyword(service: LibraryService) -> None:
    result = service.scan(keyword="linked")
    assert [item.filename for item in result.files] == ["linked_list.c"]


def test_scan_respects_max_depth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "deep"
    shallow = root / "a.py"
    root.mkdir()
    shallow.write_text(PY_SAMPLE, encoding="utf-8")
    deep_dir = root
    for index in range(6):
        deep_dir = deep_dir / f"level{index}"
    deep_dir.mkdir(parents=True)
    (deep_dir / "buried.py").write_text(PY_SAMPLE, encoding="utf-8")

    monkeypatch.setenv("LIBRARY__ROOTS", str(root))
    monkeypatch.setenv("LIBRARY__MAX_DEPTH", "2")
    get_settings.cache_clear()
    result = LibraryService(get_settings().library).scan()
    assert [item.filename for item in result.files] == ["a.py"]
    get_settings.cache_clear()


def test_scan_truncates_at_max_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "many"
    root.mkdir()
    for index in range(10):
        (root / f"f{index}.py").write_text(PY_SAMPLE, encoding="utf-8")

    monkeypatch.setenv("LIBRARY__ROOTS", str(root))
    monkeypatch.setenv("LIBRARY__MAX_FILES", "4")
    get_settings.cache_clear()
    result = LibraryService(get_settings().library).scan()
    assert result.total_files == 4
    assert result.truncated is True
    get_settings.cache_clear()


def test_oversized_file_is_listed_but_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """超大文件只登记元信息，行数留空——避免误选了几十 MB 的数据文件。"""
    root = tmp_path / "big"
    root.mkdir()
    (root / "tiny.py").write_text(PY_SAMPLE, encoding="utf-8")
    (root / "huge.py").write_text("x = 1\n" * 200, encoding="utf-8")

    monkeypatch.setenv("LIBRARY__ROOTS", str(root))
    monkeypatch.setenv("LIBRARY__MAX_FILE_BYTES", "100")
    get_settings.cache_clear()
    items = {item.filename: item for item in LibraryService(get_settings().library).scan().files}
    assert items["tiny.py"].too_large is False
    assert items["huge.py"].too_large is True
    assert items["huge.py"].line_count is None
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 读取内容
# ---------------------------------------------------------------------------
def test_read_text_returns_code_and_relative_path(service: LibraryService) -> None:
    code, rel_path, encoding, replaced = service.read_text("homework/utils.py")
    assert code == PY_SAMPLE
    assert rel_path == "homework/utils.py"
    assert encoding == "utf-8"
    assert replaced is False


def test_read_text_accepts_windows_style_separator(service: LibraryService) -> None:
    """同学从资源管理器复制路径时常常是反斜杠，要能兼容。"""
    code, rel_path, _, _ = service.read_text("homework\\utils.py")
    assert code == PY_SAMPLE
    assert rel_path == "homework/utils.py"


def test_read_text_falls_back_to_gbk(library_root: Path, service: LibraryService) -> None:
    """GBK 编码的中文注释不能直接报错——学生机上很常见。"""
    target = library_root / "homework" / "gbk.py"
    target.write_bytes("# 这是一个中文注释\nprint('你好')\n".encode("gbk"))
    code, _, encoding, replaced = service.read_text("homework/gbk.py")
    assert encoding == "gbk"
    assert "这是一个中文注释" in code
    assert replaced is False


def test_read_text_preserves_crlf(library_root: Path, service: LibraryService) -> None:
    """CRLF 换行要原样保留。

    Windows 上的学生作业几乎都是 CRLF。读取时绝不能偷偷把 \\r\\n 改成 \\n，
    否则送进 AI 的代码和ta们编辑器里看到的不是同一份，
    改错结果贴回去时还会整片显示为「全文件都被修改」。
    """
    target = library_root / "homework" / "crlf.c"
    target.write_bytes(b"int main(void) {\r\n    return 0;\r\n}\r\n")
    code, _, _, _ = service.read_text("homework/crlf.c")
    assert code == "int main(void) {\r\n    return 0;\r\n}\r\n"
    assert code.count("\r\n") == 3


def test_read_text_strips_utf8_bom(library_root: Path, service: LibraryService) -> None:
    """带 BOM 的文件要认出 utf-8-sig，并且 BOM 字符不能留在代码里。"""
    target = library_root / "homework" / "bom.py"
    target.write_bytes(b"\xef\xbb\xbf" + PY_SAMPLE.encode("utf-8"))
    code, _, encoding, _ = service.read_text("homework/bom.py")
    assert encoding == "utf-8-sig"
    assert not code.startswith("\ufeff")
    assert code == PY_SAMPLE


def test_read_text_reports_plain_utf8_for_normal_file(service: LibraryService) -> None:
    """普通 UTF-8 文件不能被误报成 utf-8-sig（utf-8-sig 解码器对无 BOM 文件也成功）。"""
    _, _, encoding, _ = service.read_text("homework/utils.py")
    assert encoding == "utf-8"


def test_read_text_rejects_traversal(service: LibraryService) -> None:
    for bad in ("../secret.py", "homework/../../secret.py", "a/../../b.py"):
        with pytest.raises(LibraryPathError):
            service.read_text(bad)


def test_read_text_rejects_absolute_path(service: LibraryService, tmp_path: Path) -> None:
    absolute = str(tmp_path / "outside.py")
    with pytest.raises(LibraryPathError):
        service.read_text(absolute)


def test_read_text_rejects_empty_path(service: LibraryService) -> None:
    with pytest.raises(LibraryPathError):
        service.read_text("   ")


def test_read_text_rejects_missing_file(service: LibraryService) -> None:
    with pytest.raises(LibraryPathError):
        service.read_text("homework/nope.py")


def test_read_text_rejects_symlink_escape(library_root: Path, service: LibraryService) -> None:
    """软链接指向根目录外时必须被挡住。

    Windows 上创建符号链接需要开发者模式或管理员权限，拿不到就跳过，
    而不是把用例删掉——在能跑的机器上它仍然守着这条边界。
    """
    outside = library_root.parent / "outside_secret.py"
    outside.write_text("SECRET = 1\n", encoding="utf-8")
    link = library_root / "homework" / "link.py"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("当前环境不支持创建符号链接，跳过该边界检查")

    with pytest.raises(LibraryPathError):
        service.read_text("homework/link.py")


def test_read_text_rejects_oversized_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "big2"
    root.mkdir()
    (root / "huge.py").write_text("x = 1\n" * 200, encoding="utf-8")
    monkeypatch.setenv("LIBRARY__ROOTS", str(root))
    monkeypatch.setenv("LIBRARY__MAX_FILE_BYTES", "100")
    get_settings.cache_clear()
    with pytest.raises(LibraryFileTooLargeError):
        LibraryService(get_settings().library).read_text("huge.py")
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def test_roots_support_csv_and_multiple_entries(tmp_path: Path) -> None:
    """`LIBRARY__ROOTS=a,b` 这种逗号写法必须能用（否则应用直接起不来）。"""
    settings = LibrarySettings(roots=f"{tmp_path / 'one'},{tmp_path / 'two'}")  # type: ignore[arg-type]
    assert settings.root_paths == [tmp_path / "one", tmp_path / "two"]


def test_roots_default_to_library_dir() -> None:
    settings = LibrarySettings()
    paths = settings.root_paths
    assert paths[0] == settings.default_path
    assert settings.default_path.name == "library"


def test_examples_are_included_by_default() -> None:
    """没配置 roots 时带上项目自带的示例目录，保证第一次打开就有东西可看。"""
    settings = LibrarySettings()
    assert settings.example_path in settings.root_paths


def test_examples_skipped_when_roots_configured_explicitly(tmp_path: Path) -> None:
    """老师配置了自己的作业目录后，不该再混进项目自带的示例。"""
    settings = LibrarySettings(roots=str(tmp_path))  # type: ignore[arg-type]
    assert settings.root_paths == [tmp_path]


def test_examples_can_be_disabled() -> None:
    settings = LibrarySettings(include_examples=False)
    assert settings.example_path not in settings.root_paths


def test_same_rel_path_in_two_roots_is_disambiguated(tmp_path: Path) -> None:
    """两个根目录下有同名文件时，必须靠 root 参数选对那一个。

    这正是 root 字段存在的理由：学生完全可能两个文件夹里各有一份 main.c。
    """
    first = tmp_path / "one"
    second = tmp_path / "two"
    for root, marker in ((first, "AAA"), (second, "BBB")):
        root.mkdir()
        (root / "main.c").write_bytes(f"/* {marker} */\nint main(void) {{ return 0; }}\n".encode())

    service = LibraryService(
        LibrarySettings(roots=f"{first},{second}")  # type: ignore[arg-type]
    )
    files = {item.root: item for item in service.scan().files}
    assert set(files) == {"one", "two"}

    code_one, _, _, _ = service.read_text("main.c", "one")
    code_two, _, _, _ = service.read_text("main.c", "two")
    assert "AAA" in code_one
    assert "BBB" in code_two


def test_unknown_root_name_is_rejected(service: LibraryService) -> None:
    with pytest.raises(LibraryPathError):
        service.read_text("homework/utils.py", "no_such_root")


def test_relative_root_resolves_against_project_root() -> None:
    from app.core.config import PROJECT_ROOT

    settings = LibrarySettings(roots="my_homework")  # type: ignore[arg-type]
    assert settings.root_paths == [PROJECT_ROOT / "my_homework"]


def test_duplicate_roots_are_deduped(tmp_path: Path) -> None:
    settings = LibrarySettings(roots=f"{tmp_path},{tmp_path}")  # type: ignore[arg-type]
    assert len(settings.root_paths) == 1


def test_settings_exposes_library_section() -> None:
    """Settings 里必须真的挂上了 library 配置，否则接口读到的是默认值。"""
    assert isinstance(Settings().library, LibrarySettings)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
def test_scan_endpoint(service: LibraryService, library_app: object) -> None:
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.get("/api/v1/library/scan")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total_files"] == 3
    assert body["language_counts"] == {"c": 1, "java": 1, "python": 1}
    # 前端靠 language_labels 生成筛选按钮
    assert body["language_labels"]["python"] == "Python"
    assert body["trace_id"]
    assert {item["language_label"] for item in body["files"]} == {"C", "Java", "Python"}


def test_scan_endpoint_language_filter(service: LibraryService, library_app: object) -> None:
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        body = client.get("/api/v1/library/scan", params={"language": "c"}).json()
    assert [item["filename"] for item in body["files"]] == ["linked_list.c"]


def test_read_file_endpoint(service: LibraryService, library_app: object) -> None:
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.get(
            "/api/v1/library/file", params={"path": "homework/utils.py"}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["code"] == PY_SAMPLE
    assert body["language"] == "python"
    assert body["language_label"] == "Python"
    assert body["line_count"] == 2


def test_read_file_endpoint_blocks_traversal(
    service: LibraryService, library_app: object
) -> None:
    """越界访问必须返回 400（请求不合法），而不是 500（服务端故障）。"""
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.get(
            "/api/v1/library/file", params={"path": "../../etc/passwd"}
        )
    assert response.status_code == 400
    assert ".." in response.json()["detail"]


def test_read_file_endpoint_missing_file_is_400(
    service: LibraryService, library_app: object
) -> None:
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.get("/api/v1/library/file", params={"path": "nope.py"})
    assert response.status_code == 400


def test_status_endpoint(service: LibraryService, library_app: object) -> None:
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        response = client.get("/api/v1/library/status")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["roots"][0]["exists"] is True
    assert body["default_path"].endswith("library")
    assert body["max_files"] > 0
    # 支持的语言里三个主打语言必须齐全
    assert {"c", "java", "python"} <= set(body["supported_languages"])


def test_library_endpoints_work_without_llm(library_app: object) -> None:
    """项目库不依赖大模型：没配 AI 时也必须完整可用。"""
    with TestClient(library_app) as client:  # type: ignore[arg-type]
        for path in ("/api/v1/library/scan", "/api/v1/library/status"):
            assert client.get(path).status_code == 200
