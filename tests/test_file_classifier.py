"""本地代码文件自动分类测试：语言识别、归档移动、安全边界、SQLite 台账、API 契约。

这个功能的每个用例都在守一件"会丢文件"的事：
覆盖同名文件、把归档目录再搬一次、跟着软链接跑到根目录外面去。
所以边界用例比正常路径还多，是刻意的。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import ClassifierSettings, LibrarySettings, Settings, get_settings
from app.core.database import dispose_engine, get_session, init_db
from app.main import create_app
from app.services.file_classifier import (
    ARCHIVE_DIRS,
    LANGUAGE_DISPLAY,
    UNKNOWN_DIR,
    UNKNOWN_LANGUAGE,
    DuplicateFileError,
    FileClassifier,
    FileClassifierError,
    FileClassifierPathError,
)
from app.services.file_record_service import FileRecordService

C_CODE = "#include <stdio.h>\n\nint main(void) { return 0; }\n"
JAVA_CODE = "public class A {\n    int f() { return 1; }\n}\n"
PY_CODE = "def add(a, b):\n    return a + b\n"
TEXT = "这是一份说明，不是代码\n"


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
@pytest.fixture()
def codes_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """造一个待分类的目录，并把配置指向它。

    一律用 write_bytes 写入（不用 write_text）：Windows 上 write_text 会把 \\n
    翻译成 \\r\\n，比对文件内容时就会对不上。
    """
    root = tmp_path / "library"
    (root / "homework").mkdir(parents=True)
    # 三种已知语言，故意放在不同层级，验证递归扫描
    (root / "main.c").write_bytes(C_CODE.encode("utf-8"))
    (root / "utils.h").write_bytes(b"int add(int a, int b);\n")
    (root / "homework" / "Calculator.java").write_bytes(JAVA_CODE.encode("utf-8"))
    (root / "homework" / "sort.py").write_bytes(PY_CODE.encode("utf-8"))
    # 未知类型：默认只登记不移动
    (root / "readme.txt").write_bytes(TEXT.encode("utf-8"))
    (root / "notes.md").write_bytes(b"# notes\n")
    # 排除目录里的代码不该被当成作业
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.c").write_bytes(C_CODE.encode("utf-8"))
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "junk.py").write_bytes(PY_CODE.encode("utf-8"))

    monkeypatch.setenv("CLASSIFIER__ROOT", str(root))
    # 归档目录也指向这里，避免动到仓库里的 data/library/
    get_settings.cache_clear()
    yield root
    get_settings.cache_clear()


@pytest.fixture()
def classifier(codes_root: Path) -> FileClassifier:
    """按当前配置构造的分类器。"""
    return FileClassifier(get_settings().classifier)


@pytest.fixture()
def files_app(sqlite_path: Path) -> Iterator[object]:
    """真实应用（分类接口不依赖大模型，无需注入假客户端）。"""
    app = create_app()
    yield app


def _run_db(scenario):
    """在一个事件循环里跑完「建表 → 执行 scenario → 释放连接池」。

    为什么必须一次跑完：SQLAlchemy 的异步引擎是和创建它的 event loop 绑定的，
    分成两次 `asyncio.run`（先建表、再查询）会报 "attached to a different loop"。
    """
    import asyncio

    async def main():
        await init_db()
        try:
            async with get_session() as session:
                return await scenario(session)
        finally:
            await dispose_engine()

    return asyncio.run(main())


# ---------------------------------------------------------------------------
# 1. 语言识别
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # 需求里的三条映射，外加大小写与 .pyi/.h 这类常见变体
        ("main.c", "c"),
        ("utils.h", "c"),
        ("MAIN.C", "c"),
        ("Calculator.java", "java"),
        ("CALC.JAVA", "java"),
        ("sort.py", "python"),
        ("types.pyi", "python"),
        # 其它一律未知
        ("readme.txt", UNKNOWN_LANGUAGE),
        ("notes.md", UNKNOWN_LANGUAGE),
        ("photo.png", UNKNOWN_LANGUAGE),
        ("Makefile", UNKNOWN_LANGUAGE),
        ("archive.tar.gz", UNKNOWN_LANGUAGE),
        # Go/JS 项目里能解析，但不在本功能要归档的三种语言内
        ("main.go", UNKNOWN_LANGUAGE),
        ("app.js", UNKNOWN_LANGUAGE),
    ],
)
def test_detect_language(filename: str, expected: str) -> None:
    assert FileClassifier.detect_language(filename) == expected


def test_unknown_note_explains_go_is_not_supported() -> None:
    """未知也要说清原因：不能让学生以为自己的 .go 文件坏了。"""
    note = FileClassifier._unknown_note("main.go")
    assert "Go" in note
    assert "未知" in note


def test_language_display_covers_four_values() -> None:
    assert LANGUAGE_DISPLAY == {
        "c": "C",
        "java": "Java",
        "python": "Python",
        "unknown": "未知",
    }


def test_archive_dirs_match_requirements() -> None:
    """归档目录名必须与需求一致：c/ java/ python/。"""
    assert ARCHIVE_DIRS == {"c": "c", "java": "java", "python": "python"}
    assert UNKNOWN_DIR == "unknown"


# ---------------------------------------------------------------------------
# 2. 目录解析与安全边界
# ---------------------------------------------------------------------------
def test_resolve_root_defaults_to_configured_root(classifier: FileClassifier, codes_root: Path) -> None:
    assert classifier.resolve_root() == codes_root.resolve()
    assert classifier.resolve_root("") == codes_root.resolve()


def test_resolve_root_accepts_subdirectory(classifier: FileClassifier, codes_root: Path) -> None:
    assert classifier.resolve_root("homework") == (codes_root / "homework").resolve()


def test_resolve_root_accepts_absolute_path_inside_root(
    classifier: FileClassifier, codes_root: Path
) -> None:
    assert classifier.resolve_root(str(codes_root / "homework")) == (
        codes_root / "homework"
    ).resolve()


@pytest.mark.parametrize("bad", ["../outside", "homework/../..", ".."])
def test_resolve_root_rejects_traversal(
    classifier: FileClassifier, codes_root: Path, bad: str
) -> None:
    """越界路径必须被拦住 —— 这个接口不能变成"任意路径搬运工"。"""
    with pytest.raises(FileClassifierPathError):
        classifier.resolve_root(bad)


def test_resolve_root_rejects_absolute_path_outside(
    classifier: FileClassifier, tmp_path: Path
) -> None:
    with pytest.raises(FileClassifierPathError):
        classifier.resolve_root(str(tmp_path))


def test_resolve_root_rejects_missing_directory(classifier: FileClassifier) -> None:
    with pytest.raises(FileClassifierPathError):
        classifier.resolve_root("no_such_dir")


def test_resolve_root_rejects_file(classifier: FileClassifier, codes_root: Path) -> None:
    with pytest.raises(FileClassifierPathError):
        classifier.resolve_root("main.c")


# ---------------------------------------------------------------------------
# 3. 扫描与归档
# ---------------------------------------------------------------------------
def test_classify_moves_code_into_language_dirs(classifier: FileClassifier, codes_root: Path) -> None:
    """核心行为：.c/.h 进 c/，.java 进 java/，.py 进 python/。"""
    classifier.classify()

    assert (codes_root / "c" / "main.c").is_file()
    assert (codes_root / "c" / "utils.h").is_file()
    assert (codes_root / "java" / "Calculator.java").is_file()
    assert (codes_root / "python" / "sort.py").is_file()

    # 原位置必须已经不在了（是移动，不是复制）
    assert not (codes_root / "main.c").exists()
    assert not (codes_root / "homework" / "Calculator.java").exists()
    assert not (codes_root / "homework" / "sort.py").exists()

    # 内容要原样保留
    assert (codes_root / "c" / "main.c").read_text(encoding="utf-8") == C_CODE


def test_classify_counts_and_actions(classifier: FileClassifier) -> None:
    report = classifier.classify()

    assert report.counts == {"c": 2, "python": 1, "java": 1, "unknown": 2}
    assert report.total == 6          # node_modules / __pycache__ 里的不算
    assert report.moved == 4          # 三种已知语言各就各位
    assert report.unknown == 2
    assert report.skipped == 0
    assert report.dry_run is False


def test_unknown_files_are_recorded_but_not_moved(classifier: FileClassifier, codes_root: Path) -> None:
    """未知文件默认留在原地（只登记），避免把别的重要资料搬走。"""
    report = classifier.classify()

    kept = [item for item in report.files if item.action == "kept"]
    assert {item.filename for item in kept} == {"readme.txt", "notes.md"}
    assert (codes_root / "readme.txt").is_file()
    assert not (codes_root / UNKNOWN_DIR).exists()
    assert all(item.note for item in kept), "未移动的文件必须说明原因"


def test_move_unknown_moves_them_into_unknown_dir(classifier: FileClassifier, codes_root: Path) -> None:
    report = classifier.classify(move_unknown=True)

    assert (codes_root / UNKNOWN_DIR / "readme.txt").is_file()
    assert (codes_root / UNKNOWN_DIR / "notes.md").is_file()
    assert report.counts[UNKNOWN_LANGUAGE] == 2


def test_excluded_dirs_are_skipped(classifier: FileClassifier, codes_root: Path) -> None:
    """node_modules / __pycache__ 里的代码不是学生作业，不能搬进来。"""
    classifier.classify()
    assert not (codes_root / "c" / "dep.c").exists()
    assert not (codes_root / "python" / "junk.py").exists()


def test_classify_does_not_mutate_unknown_contents(classifier: FileClassifier, codes_root: Path) -> None:
    classifier.classify()
    assert (codes_root / "readme.txt").read_text(encoding="utf-8") == TEXT


# ---------------------------------------------------------------------------
# 4. 幂等 / 重名 / 上限 —— 三条"别丢文件"的底线
# ---------------------------------------------------------------------------
def test_second_scan_finds_nothing_to_move(classifier: FileClassifier) -> None:
    """重复扫描必须没有副作用：归档目录会被跳过，不会出现 c/c/ 这种套娃。"""
    first = classifier.classify()
    assert first.moved == 4

    second = classifier.classify()
    assert second.total == 2, "第二次只剩两个未知文件（归档目录被跳过）"
    assert second.moved == 0


def test_second_scan_does_not_nest_archive_dirs(classifier: FileClassifier, codes_root: Path) -> None:
    classifier.classify()
    classifier.classify()
    assert not (codes_root / "c" / "c").exists()
    assert sorted(p.name for p in (codes_root / "c").iterdir()) == ["main.c", "utils.h"]


def test_duplicate_names_get_suffix_and_nothing_is_lost(
    classifier: FileClassifier, codes_root: Path
) -> None:
    """两个子目录里各有一个 main.c 时，绝不能覆盖，必须加后缀且内容都还在。"""
    (codes_root / "other").mkdir()
    (codes_root / "other" / "main.c").write_bytes(b"/* second main */\n")

    report = classifier.classify()

    archive = codes_root / "c"
    assert (archive / "main.c").is_file()
    assert (archive / "main_1.c").is_file()
    contents = {
        (archive / "main.c").read_text(encoding="utf-8"),
        (archive / "main_1.c").read_text(encoding="utf-8"),
    }
    assert contents == {C_CODE, "/* second main */\n"}
    # 被改名的那个文件必须明确告知，否则学生会以为文件丢了
    renamed = [item for item in report.files if item.filename == "main.c" and item.target_path.endswith("main_1.c")]
    assert renamed and "重命名" in renamed[0].note


def test_archive_dir_files_are_left_alone(classifier: FileClassifier, codes_root: Path) -> None:
    """已经在 c/ 里的文件不应该被再次处理（也不会被移到别处）。"""
    (codes_root / "c").mkdir(exist_ok=True)
    (codes_root / "c" / "old.c").write_bytes(C_CODE.encode("utf-8"))

    report = classifier.classify()

    assert (codes_root / "c" / "old.c").is_file()
    assert all(item.filename != "old.c" for item in report.files)


def test_nested_dir_named_like_archive_is_scanned(classifier: FileClassifier, codes_root: Path) -> None:
    """只有**根目录下**的 c/ java/ python/ 才算归档目录；src/c/ 是正常代码目录。"""
    (codes_root / "src" / "c").mkdir(parents=True)
    (codes_root / "src" / "c" / "helper.c").write_bytes(C_CODE.encode("utf-8"))

    classifier.classify()

    assert (codes_root / "c" / "helper.c").is_file()


def test_non_recursive_only_scans_top_level(classifier: FileClassifier, codes_root: Path) -> None:
    report = classifier.classify(recursive=False)

    # 只处理根目录这一层：main.c、utils.h、readme.txt、notes.md
    assert report.total == 4
    assert (codes_root / "c" / "main.c").is_file()
    assert (codes_root / "homework" / "Calculator.java").is_file(), "子目录不该被碰"


def test_max_files_truncates(classifier: FileClassifier) -> None:
    report = classifier.classify(max_files=2)
    assert report.total == 2
    assert report.truncated is True


def test_oversized_file_is_skipped(
    classifier: FileClassifier, codes_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (codes_root / "big.c").write_bytes(b"x" * 2048)
    monkeypatch.setenv("CLASSIFIER__MAX_FILE_BYTES", "1000")
    get_settings.cache_clear()
    sized = FileClassifier(get_settings().classifier)

    report = sized.classify()

    skipped = [item for item in report.files if item.action == "skipped"]
    assert [item.filename for item in skipped] == ["big.c"]
    assert "超过上限" in skipped[0].note
    assert (codes_root / "big.c").is_file(), "跳过的文件必须留在原地"


def test_symlink_is_skipped(classifier: FileClassifier, codes_root: Path, tmp_path: Path) -> None:
    """软链接可能指向根目录之外，必须跳过（Windows 需要开发者模式，拿不到就跳过用例）。"""
    outside = tmp_path / "outside.c"
    outside.write_bytes(C_CODE.encode("utf-8"))
    link = codes_root / "link.c"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("当前环境不支持创建符号链接，跳过该边界检查")

    report = classifier.classify()

    assert all(item.filename != "link.c" for item in report.files)
    assert link.is_symlink(), "软链接本身不能被搬走或破坏"
    assert outside.is_file(), "链接指向的目标文件必须原封不动"


# ---------------------------------------------------------------------------
# 5. 预演模式
# ---------------------------------------------------------------------------
def test_dry_run_plans_without_moving(classifier: FileClassifier, codes_root: Path) -> None:
    report = classifier.classify(dry_run=True)

    assert report.dry_run is True
    assert report.moved == 4
    assert all(item.action == "planned" for item in report.files if item.language != UNKNOWN_LANGUAGE)
    # 磁盘上什么都没动
    assert (codes_root / "main.c").is_file()
    assert (codes_root / "homework" / "sort.py").is_file()
    assert not (codes_root / "c").exists() or not (codes_root / "c" / "main.c").exists()


def test_dry_run_reports_planned_targets(classifier: FileClassifier) -> None:
    report = classifier.classify(dry_run=True)
    planned = {item.filename: item.target_path for item in report.files if item.action == "planned"}

    assert planned["main.c"] == "c/main.c"
    assert planned["utils.h"] == "c/utils.h"
    assert planned["Calculator.java"] == "java/Calculator.java"
    assert planned["sort.py"] == "python/sort.py"


# ---------------------------------------------------------------------------
# 6. SQLite 台账（服务层直连，不经接口）
# ---------------------------------------------------------------------------
def test_save_report_writes_metadata(classifier: FileClassifier, sqlite_path: Path) -> None:
    """需求点：记录文件名、语言、路径、大小、上传时间。"""
    report = classifier.classify()

    async def scenario(session):
        inserted, updated = await FileRecordService.save_report(session, report)
        items, total = await FileRecordService.list_files(session)
        return inserted, updated, items, total

    inserted, updated, items, total = _run_db(scenario)

    assert (inserted, updated) == (6, 0)   # 4 个已归档 + 2 个未知
    assert total == 6

    record = next(item for item in items if item.filename == "sort.py")
    assert record.language == "python"
    assert record.path == "python/sort.py"
    assert record.size_bytes == len(PY_CODE.encode("utf-8"))
    assert record.created_at is not None      # 「上传时间」
    assert record.archived is True
    assert record.file_modified_at is not None


def test_save_report_is_idempotent(classifier: FileClassifier, sqlite_path: Path) -> None:
    """重复扫描只更新不新增 —— 否则列表会堆满同一份文件的历史记录。"""
    report = classifier.classify()

    async def scenario(session):
        await FileRecordService.save_report(session, report)
        # 第二次用同样的报告再存一遍（模拟重复扫描）
        inserted, updated = await FileRecordService.save_report(session, report)
        _, total = await FileRecordService.list_files(session)
        return inserted, updated, total

    inserted, updated, total = _run_db(scenario)
    assert inserted == 0
    assert updated == 6
    assert total == 6


def test_dry_run_writes_nothing(classifier: FileClassifier, sqlite_path: Path) -> None:
    report = classifier.classify(dry_run=True)

    async def scenario(session):
        inserted, updated = await FileRecordService.save_report(session, report)
        _, total = await FileRecordService.list_files(session)
        return inserted, updated, total

    assert _run_db(scenario) == (0, 0, 0)


def test_skipped_files_are_not_recorded(
    classifier: FileClassifier,
    codes_root: Path,
    sqlite_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (codes_root / "huge.c").write_bytes(b"x" * 2048)
    monkeypatch.setenv("CLASSIFIER__MAX_FILE_BYTES", "1000")
    get_settings.cache_clear()
    report = FileClassifier(get_settings().classifier).classify()

    async def scenario(session):
        await FileRecordService.save_report(session, report)
        items, _ = await FileRecordService.list_files(session)
        return [item.filename for item in items]

    assert "huge.c" not in _run_db(scenario)


def test_list_files_filters_and_counts(classifier: FileClassifier, sqlite_path: Path) -> None:
    report = classifier.classify()

    async def scenario(session):
        await FileRecordService.save_report(session, report)
        only_c, total_c = await FileRecordService.list_files(session, language="c")
        counts = await FileRecordService.count_by_language(session)
        page, _ = await FileRecordService.list_files(session, limit=2)
        return only_c, total_c, counts, page

    only_c, total_c, counts, page = _run_db(scenario)
    assert total_c == 2
    assert {item.filename for item in only_c} == {"main.c", "utils.h"}
    assert counts == {"c": 2, "python": 1, "java": 1, "unknown": 2}
    assert len(page) == 2


# ---------------------------------------------------------------------------
# 7. API
# ---------------------------------------------------------------------------
def test_scan_endpoint(classifier: FileClassifier, files_app: object, codes_root: Path) -> None:
    with TestClient(files_app) as client:  # type: ignore[arg-type]
        response = client.post("/api/v1/files/scan", json={})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 6
    assert body["moved"] == 4
    assert body["unknown"] == 2
    assert body["counts"] == {"c": 2, "python": 1, "java": 1, "unknown": 2}
    assert body["language_labels"]["c"] == "C"
    assert body["language_labels"]["unknown"] == "未知"
    assert body["inserted"] == 6
    assert body["trace_id"]
    assert (codes_root / "c" / "main.c").is_file()


def test_scan_endpoint_works_without_body(files_app: object) -> None:
    """不传请求体也应该能用（等价于全默认参数）。"""
    with TestClient(files_app) as client:  # type: ignore[arg-type]
        response = client.post("/api/v1/files/scan")
    assert response.status_code == 200, response.text


def test_scan_endpoint_dry_run(files_app: object, codes_root: Path) -> None:
    with TestClient(files_app) as client:  # type: ignore[arg-type]
        response = client.post("/api/v1/files/scan", json={"dry_run": True})

    body = response.json()
    assert body["dry_run"] is True
    assert body["inserted"] == 0
    assert (codes_root / "main.c").is_file(), "预演不能移动文件"


def test_scan_endpoint_subdirectory(files_app: object, codes_root: Path) -> None:
    with TestClient(files_app) as client:  # type: ignore[arg-type]
        response = client.post("/api/v1/files/scan", json={"directory": "homework"})

    body = response.json()
    assert body["total"] == 2
    assert (codes_root / "java" / "Calculator.java").is_file()
    assert (codes_root / "main.c").is_file(), "根目录的文件不该被这次扫描碰到"


def test_scan_endpoint_rejects_traversal(files_app: object) -> None:
    with TestClient(files_app) as client:  # type: ignore[arg-type]
        response = client.post("/api/v1/files/scan", json={"directory": "../.."})

    assert response.status_code == 400
    assert "只能扫描" in response.json()["detail"] or "不存在" in response.json()["detail"]


def test_scan_endpoint_move_unknown(files_app: object, codes_root: Path) -> None:
    with TestClient(files_app) as client:  # type: ignore[arg-type]
        client.post("/api/v1/files/scan", json={"move_unknown": True})

    assert (codes_root / "unknown" / "readme.txt").is_file()


def test_list_endpoint_after_scan(files_app: object, codes_root: Path) -> None:
    with TestClient(files_app) as client:  # type: ignore[arg-type]
        client.post("/api/v1/files/scan", json={})
        response = client.get("/api/v1/files/list")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 6
    assert body["language"] is None
    assert body["by_language"] == {"c": 2, "python": 1, "java": 1, "unknown": 2}

    item = next(row for row in body["items"] if row["filename"] == "main.c")
    assert item["language"] == "c"
    assert item["language_label"] == "C"
    assert item["path"] == "c/main.c"
    assert item["archived"] is True
    assert item["created_at"]
    assert item["exists"] is True


def test_list_endpoint_filters_by_language(files_app: object, codes_root: Path) -> None:
    with TestClient(files_app) as client:  # type: ignore[arg-type]
        client.post("/api/v1/files/scan", json={})
        response = client.get("/api/v1/files/list", params={"language": "java"})

    body = response.json()
    assert body["language"] == "java"
    assert body["total"] == 1
    assert [row["filename"] for row in body["items"]] == ["Calculator.java"]


def test_list_endpoint_rejects_unknown_language(files_app: object, codes_root: Path) -> None:
    with TestClient(files_app) as client:  # type: ignore[arg-type]
        response = client.get("/api/v1/files/list", params={"language": "cobol"})
    assert response.status_code == 400
    assert "不支持的语言" in response.json()["detail"]


def test_list_endpoint_marks_missing_files(files_app: object, codes_root: Path) -> None:
    """文件在程序外面被删掉时，列表要如实标注，而不是假装它还在。"""
    with TestClient(files_app) as client:  # type: ignore[arg-type]
        client.post("/api/v1/files/scan", json={})
        (codes_root / "c" / "main.c").unlink()
        response = client.get("/api/v1/files/list", params={"language": "c"})

    items = {row["filename"]: row for row in response.json()["items"]}
    assert items["main.c"]["exists"] is False
    assert items["utils.h"]["exists"] is True


def test_list_endpoint_pagination(files_app: object, codes_root: Path) -> None:
    with TestClient(files_app) as client:  # type: ignore[arg-type]
        client.post("/api/v1/files/scan", json={})
        first = client.get("/api/v1/files/list", params={"limit": 2}).json()
        second = client.get("/api/v1/files/list", params={"limit": 2, "offset": 2}).json()

    assert len(first["items"]) == 2
    assert len(second["items"]) == 2
    assert {row["id"] for row in first["items"]} & {row["id"] for row in second["items"]} == set()


def test_empty_directory_scans_cleanly(tmp_path: Path, files_app: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """空目录不是错误：返回 0 个文件，而不是 500。"""
    empty = tmp_path / "empty_codes"
    empty.mkdir()
    monkeypatch.setenv("CLASSIFIER__ROOT", str(empty))
    get_settings.cache_clear()

    with TestClient(files_app) as client:  # type: ignore[arg-type]
        response = client.post("/api/v1/files/scan", json={})

    assert response.status_code == 200
    assert response.json()["total"] == 0
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 8. 前端拖拽入库（save_upload + /files/upload）
# ---------------------------------------------------------------------------
def test_save_upload_writes_file_to_root(classifier: FileClassifier, codes_root: Path) -> None:
    """拖进来的文件要落到分类根目录，等 /files/scan 归档。"""
    result = classifier.save_upload("newby.py", b"print('hello')\n")

    assert result.saved is True
    assert result.filename == "newby.py"
    assert result.language == "python"
    assert result.relative_path == "newby.py"
    assert (codes_root / "newby.py").read_bytes() == b"print('hello')\n"


@pytest.mark.parametrize(
    ("name", "language"),
    [("a.c", "c"), ("a.h", "c"), ("A.java", "java"), ("a.py", "python")],
)
def test_save_upload_accepts_supported_languages(
    classifier: FileClassifier, name: str, language: str
) -> None:
    assert classifier.save_upload(name, b"x = 1\n").language == language


@pytest.mark.parametrize("name", ["notes.txt", "app.go", "archive.zip", "README.md"])
def test_save_upload_rejects_unsupported_suffix(classifier: FileClassifier, name: str) -> None:
    """项目库只收三种语言的源码：拖进来其它文件要立刻拒绝并说清原因。"""
    with pytest.raises(FileClassifierPathError) as excinfo:
        classifier.save_upload(name, b"content")
    assert "只支持" in str(excinfo.value)


@pytest.mark.parametrize("name", ["", "   ", ".", ".."])
def test_save_upload_rejects_bad_name(classifier: FileClassifier, name: str) -> None:
    with pytest.raises(FileClassifierPathError):
        classifier.save_upload(name, b"x = 1\n")


def test_save_upload_strips_path_from_filename(
    classifier: FileClassifier, codes_root: Path
) -> None:
    """文件名里的路径必须被丢掉——否则 `../../evil.py` 就能写到仓库外面去。"""
    result = classifier.save_upload("../../../evil.py", b"x = 1\n")

    assert result.filename == "evil.py"
    assert (codes_root / "evil.py").is_file()
    # 仓库根目录之外不该出现这个文件
    assert not (codes_root.parent.parent / "evil.py").exists()


def test_save_upload_rejects_empty_and_oversized(
    classifier: FileClassifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FileClassifierError):
        classifier.save_upload("empty.py", b"")

    monkeypatch.setenv("CLASSIFIER__MAX_FILE_BYTES", "1000")
    get_settings.cache_clear()
    sized = FileClassifier(get_settings().classifier)
    with pytest.raises(FileClassifierError) as excinfo:
        sized.save_upload("big.py", b"x" * 2048)
    assert "超过上限" in str(excinfo.value)
    get_settings.cache_clear()


def test_save_upload_detects_duplicate(classifier: FileClassifier) -> None:
    """同名 + 内容相同 → 报"文件已存在"（前端据此提示，不重复添加）。"""
    classifier.save_upload("same.py", b"print(1)\n")

    with pytest.raises(DuplicateFileError) as excinfo:
        classifier.save_upload("same.py", b"print(1)\n")

    assert "文件已存在" in str(excinfo.value)
    assert excinfo.value.existing_path == "same.py"


def test_save_upload_detects_duplicate_in_archive_dir(classifier: FileClassifier) -> None:
    """文件被归档到 python/ 之后，再拖同一个文件依然要能判成重复。

    回归：如果只查根目录，第一次归档后就永远查不出重复，
    学生会看到项目库里莫名多出 main_1.py、main_2.py……
    """
    classifier.save_upload("dup.py", b"print(2)\n")
    classifier.classify()                       # 归档到 python/dup.py
    assert (classifier.settings.root_path / "python" / "dup.py").is_file()

    with pytest.raises(DuplicateFileError) as excinfo:
        classifier.save_upload("dup.py", b"print(2)\n")
    assert "python/dup.py" in excinfo.value.existing_path


def test_save_upload_renames_when_content_differs(classifier: FileClassifier) -> None:
    """同名但内容不同 → 另存为新名字，绝不覆盖（两个学生可能都叫 main.c）。

    注意用一个夹具里不存在的文件名，序列才是确定的：
    第一次是全新文件（不改名），第二次同名不同内容才触发改名。
    """
    first = classifier.save_upload("answer.c", b"int main(void) { return 0; }\n")
    second = classifier.save_upload("answer.c", b"int main(void) { return 1; }\n")

    assert first.filename == "answer.c" and first.renamed is False
    assert second.renamed is True
    assert second.filename == "answer_1.c"
    assert "另存为" in second.note

    root = classifier.settings.root_path
    # 两份内容都在，谁都没被覆盖
    assert (root / "answer.c").read_bytes() == b"int main(void) { return 0; }\n"
    assert (root / "answer_1.c").read_bytes() == b"int main(void) { return 1; }\n"


def test_save_upload_renames_against_archived_name(classifier: FileClassifier) -> None:
    """归档目录里已有同名文件时，新上传的也要改名（回归）。

    真实场景：学生拖入 homework.py（已归档到 python/homework.py），
    改了内容再拖一次。此时根目录看起来"没有同名文件"，
    如果只比较根目录就会写出第二个 homework.py，列表里出现两个同名文件，
    下一次归档还会再撞名改一次名——所以要按**整个项目库**判重。
    """
    root = classifier.settings.root_path
    classifier.save_upload("again.py", b"print('v1')\n")
    classifier.classify()                          # 归档到 python/again.py
    assert (root / "python" / "again.py").is_file()

    result = classifier.save_upload("again.py", b"print('v2')\n")

    assert result.renamed is True
    assert result.filename == "again_1.py"
    # 两份内容都还在，谁都没被覆盖
    assert (root / "python" / "again.py").read_bytes() == b"print('v1')\n"
    assert (root / "again_1.py").read_bytes() == b"print('v2')\n"


def test_uploaded_file_shows_up_in_library_scan(
    classifier: FileClassifier, codes_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """拖进来的文件必须能被「本地项目库」扫描到，否则列表上看不到，像没交上。

    这是"分类器根目录与项目库根目录必须是同一个"的回归用例。
    """
    from app.services.library_service import LibraryService

    monkeypatch.setenv("LIBRARY__ROOTS", str(codes_root))
    get_settings.cache_clear()

    classifier.save_upload("homework_drop.py", b"print('dropped')\n")
    library = LibraryService(get_settings().library)
    files = [item.rel_path for item in library.scan().files]

    assert "homework_drop.py" in files
    get_settings.cache_clear()


def test_uploaded_file_found_after_archiving(
    classifier: FileClassifier, codes_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """归档之后（文件被移到 python/ 下）项目库依然能看到，并且能读回内容。"""
    from app.services.library_service import LibraryService

    monkeypatch.setenv("LIBRARY__ROOTS", str(codes_root))
    get_settings.cache_clear()

    classifier.save_upload("archived.py", b"print('archived')\n")
    report = classifier.classify()
    assert report.moved >= 1

    library = LibraryService(get_settings().library)
    paths = [item.rel_path for item in library.scan().files]
    assert "python/archived.py" in paths

    code, rel_path, _, _ = library.read_text("python/archived.py")
    assert code == "print('archived')\n"
    assert rel_path == "python/archived.py"


def test_classifier_skips_library_trash_dir(
    classifier: FileClassifier, codes_root: Path
) -> None:
    """分类器必须跳过项目库的回收站 `.trash`，不能把删掉的文件搬回来。

    回归用例（真实使用中踩到的）：学生删掉一个作业 → 文件进了 `.trash/`；*
    接着往项目库拖另一个文件 → 这次归档递归扫描时走进了 `.trash/`，
    把刚删掉的文件又搬回 `python/`，名字还带着回收站的时间戳前缀
    （`20260918-160000__python__build_report.py`）。学生看到"删了还在"，一头雾水。
    """
    (codes_root / ".trash").mkdir()
    trashed = codes_root / ".trash" / "20260918-160000__python__deleted.py"
    trashed.write_bytes(PY_CODE.encode("utf-8"))

    report = classifier.classify()

    # 回收站里的文件既不该被移动，也不该出现在本次分类结果里
    assert trashed.exists(), "回收站里的文件被分类器动过了"
    assert not (codes_root / "python" / trashed.name).exists(), "删掉的文件又被搬回 python/ 了"
    assert all(trashed.name not in item.filename for item in report.files)


def test_classifier_skips_custom_trash_dir_name(
    codes_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """回收站目录名可配置时同样要跳过（配置自己会把名字补进分类器黑名单）。

    否则一旦有人把 `LIBRARY__TRASH_DIR_NAME` 改成 `recycle`，
    分类器的黑名单还是只有 `.trash`，上面那个"删了又回来"的 bug 会原地复发。
    """
    monkeypatch.setenv("LIBRARY__TRASH_DIR_NAME", "recycle")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert "recycle" in settings.classifier.excluded_dirs

        custom_trash = codes_root / "recycle"
        custom_trash.mkdir()
        trashed = custom_trash / "old_homework.py"
        trashed.write_bytes(PY_CODE.encode("utf-8"))

        FileClassifier(settings.classifier).classify()

        assert trashed.exists(), "自定义回收站里的文件被搬走了"
        assert not (codes_root / "python" / "old_homework.py").exists()
    finally:
        get_settings.cache_clear()


def test_default_excluded_dirs_cover_trash() -> None:
    """两边配置的默认值都要带上 `.trash`：库扫描与分类器各有一份清单。

    这条断言很便宜，但能挡住"以后有人整理配置时删掉一行"的回归。
    """
    assert ".trash" in ClassifierSettings().excluded_dirs
    assert ".trash" in LibrarySettings().excluded_dirs

    get_settings.cache_clear()


def test_upload_endpoint_flow(files_app: object, codes_root: Path) -> None:
    """接口全流程：拖入 → /upload 存盘 → /scan 归档 → 列表与台账都能看到。"""
    from fastapi.testclient import TestClient

    with TestClient(files_app) as client:  # type: ignore[arg-type]
        # 1) 上传
        response = client.post(
            "/api/v1/files/upload",
            files={"file": ("dropped.py", b"print('hi')\n", "text/x-python")},
        )
        assert response.status_code == 201, response.text
        uploaded = response.json()
        assert uploaded["filename"] == "dropped.py"
        assert uploaded["language"] == "python"
        assert uploaded["renamed"] is False
        assert uploaded["trace_id"]

        # 2) 归档 + 写台账
        assert client.post("/api/v1/files/scan", json={}).status_code == 200
        listing = client.get("/api/v1/files/list").json()
        names = {item["filename"] for item in listing["items"]}
        assert "dropped.py" in names

        # 3) 归档后的位置也在台账里（python/dropped.py）
        record = next(item for item in listing["items"] if item["filename"] == "dropped.py")
        assert record["path"] == "python/dropped.py"
        assert record["language"] == "python"
        assert record["size_bytes"] == len(b"print('hi')\n")


def test_upload_endpoint_duplicate_returns_409(files_app: object) -> None:
    """重复拖入同一个文件 → 409 + "文件已存在"（前端据此提示，不重复添加）。"""
    from fastapi.testclient import TestClient

    with TestClient(files_app) as client:  # type: ignore[arg-type]
        payload = {"file": ("same.py", b"print(1)\n", "text/x-python")}
        assert client.post("/api/v1/files/upload", files=payload).status_code == 201

        again = client.post("/api/v1/files/upload", files=payload)
        assert again.status_code == 409
        assert "文件已存在" in again.json()["detail"]


def test_upload_endpoint_rejects_bad_suffix(files_app: object) -> None:
    from fastapi.testclient import TestClient

    with TestClient(files_app) as client:  # type: ignore[arg-type]
        response = client.post(
            "/api/v1/files/upload", files={"file": ("a.txt", b"hello", "text/plain")}
        )
    assert response.status_code == 400
    assert "只支持" in response.json()["detail"]


def test_upload_endpoint_renames_same_name(files_app: object) -> None:
    """同名不同内容 → 201 + renamed=true（不覆盖已有文件）。"""
    from fastapi.testclient import TestClient

    with TestClient(files_app) as client:  # type: ignore[arg-type]
        client.post("/api/v1/files/upload", files={"file": ("m.py", b"print(1)\n", "text/plain")})
        response = client.post(
            "/api/v1/files/upload", files={"file": ("m.py", b"print(2)\n", "text/plain")}
        )

    assert response.status_code == 201
    assert response.json()["renamed"] is True
    assert response.json()["filename"] == "m_1.py"


# ---------------------------------------------------------------------------
# 9. 配置
# ---------------------------------------------------------------------------
def test_settings_exposes_classifier_section() -> None:
    """Settings 里必须真的挂上 classifier 配置，否则接口读到的是默认值。"""
    assert isinstance(Settings().classifier, ClassifierSettings)


def test_relative_root_resolves_against_project_root() -> None:
    from app.core.config import PROJECT_ROOT

    assert ClassifierSettings(root="data/library").root_path == PROJECT_ROOT / "data" / "library"  # type: ignore[arg-type]


def test_default_root_is_shared_with_library() -> None:
    """分类根目录默认与「本地项目库」用同一个文件夹。

    回归（这条不变量坏掉会让"拖进项目库"看起来失效）：
    前端拖入文件的流程是 上传 → /files/scan 归档 → 刷新左侧列表，
    而左侧列表来自 /library/scan。两个功能必须指向同一个目录，
    否则刚拖进去的文件不会出现在列表里，学生会以为作业没交上。
    """
    classifier = ClassifierSettings()
    library = LibrarySettings()

    assert classifier.root_path == library.default_path
    assert classifier.root_path.name == "library"


def test_excluded_dirs_support_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """用户配置的黑名单要支持逗号分隔写法，同时**始终**带上回收站目录。

    末尾那个 `.trash` 是配置层自动补上的（见 Settings._protect_library_trash_from_classifier）：
    回收站必须永远排除在分类器之外，否则"删掉的文件又被搬回来"那个 bug 会复发。
    所以这里断言的是"用户写的三项 + 自动补的回收站"，而不是原来的三项。
    """
    monkeypatch.setenv("CLASSIFIER__EXCLUDED_DIRS", "a,b , c")
    get_settings.cache_clear()
    assert get_settings().classifier.excluded_dirs == ["a", "b", "c", ".trash"]
    get_settings.cache_clear()


def test_ensure_directories_creates_archive_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """启动时就把 c/ java/ python/ 摆好，学生一看就知道文件会归到哪。"""
    root = tmp_path / "codes"
    monkeypatch.setenv("CLASSIFIER__ROOT", str(root))
    get_settings.cache_clear()
    get_settings().ensure_directories()

    assert (root / "c").is_dir()
    assert (root / "java").is_dir()
    assert (root / "python").is_dir()
    get_settings.cache_clear()
