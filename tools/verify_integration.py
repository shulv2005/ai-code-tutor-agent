"""整合自检：确认所有模块、依赖、文档、脚本都真的接上了。

为什么需要这个脚本：
整合类的工作最容易出"静默漏掉"——某个模块忘了挂路由（访问时 404）、
requirements.txt 少了一个包（换台机器就装不起来）、README 里写的接口
根本不存在（学生照着敲却调不通）。这些都不会报错，只会让人在课堂上卡住。

本脚本逐条检查这些"不说话的问题"：
  1. 路由注册完整性：各模块的标签与接口数是否符合预期
  2. 依赖清单一致性：requirements.txt ↔ pyproject.toml
  3. README 里列出的每个接口路径都真实存在
  4. README 是否包含需求要求的 5 个章节
  5. 前端页面与静态资源是否齐全
  6. .bat 脚本存在、编码正确、且打印了正确的页面地址
  7. 数据库的 7 张表都能建出来
  8. 各模块的 service / endpoint 文件都在

运行：python tools/verify_integration.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# 直接 `python tools/verify_integration.py` 运行时，sys.path[0] 是 tools/ 而不是项目根目录，
# 于是 `import app` 会报 ModuleNotFoundError。这里把项目根目录补进去，
# 让脚本无论从哪个目录、用哪种方式调用都能跑起来（踩过一次，别再让使用者踩）。
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# 期望的模块标签 -> 最少接口数。
# 用途：任何模块被误删或漏挂都会在这里暴露。
# 数字只写"下限"，以后加接口不用改这个表；但少一个模块立刻失败。
EXPECTED_MODULES: dict[str, int] = {
    "health": 2,
    "repositories": 7,
    "retrieval": 1,
    "agent": 3,
    "sandbox": 2,
    "tutor": 8,
    # 项目库现在有 5 个操作：scan / file(GET) / file(PUT) / file(DELETE) / status。
    # 写两个（PUT/DELETE）是「在线编辑保存 / 替换 / 删除」用的，
    # 谁把它们误删了，这条断言会立刻报警。
    "library": 5,
    "files": 2,
    "check": 1,
    "fix": 3,
    "comment": 3,
    # 多模型与 API Key 会话：set_key / clear_key / status
    "auth": 3,
    # 模型清单：前端下拉框的数据源（不含 Key）
    "models": 1,
}

# 期望存在的数据库表（每个功能模块的落库表）
EXPECTED_TABLES = {
    "repositories",
    "code_files",
    "code_symbols",
    "tutor_records",
    "classified_files",
    "code_check_records",
    "code_fix_records",
    "comment_records",
}

# README 必须包含的章节关键词（对应需求里点名的内容）
EXPECTED_README_SECTIONS = [
    "项目功能",
    "一键启动",
    "如何使用 HTML 界面",
    "如何添加学生代码",
    "AI 功能的原理",
]

# 前端必须存在的文件
EXPECTED_FRONTEND_FILES = [
    "frontend/index.html",
    "frontend/css/style.css",
    "frontend/js/app.js",
    "frontend/vendor/highlight.min.js",
    "frontend/vendor/github.min.css",
]

# 各模块的核心实现文件（少一个说明整合不完整）
EXPECTED_SERVICE_FILES = [
    "app/api/v1/router.py",
    "app/api/v1/endpoints/check.py",
    "app/api/v1/endpoints/fix.py",
    "app/api/v1/endpoints/comment.py",
    "app/api/v1/endpoints/files.py",
    "app/api/v1/endpoints/library.py",
    "app/api/v1/endpoints/tutor.py",
    "app/services/code_checker.py",
    "app/services/code_fixer.py",
    "app/services/comment_generator.py",
    "app/services/classifier_placeholder",  # 占位：下面会单独判断 file_classifier
]

checks: list[tuple[str, bool]] = []


def report(label: str, ok: bool, detail: str = "") -> None:
    """记一条检查结果并打印。"""
    checks.append((label, ok))
    mark = "OK  " if ok else "FAIL"
    print(f"  {mark} {label}" + (f"  —— {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 1. 路由注册完整性
# ---------------------------------------------------------------------------
def check_routes() -> set[str]:
    """检查所有模块是否都挂上了，返回已注册的路径集合。"""
    print("=" * 72)
    print("1. 路由注册完整性")
    print("=" * 72)

    from app.main import create_app, frontend_page_urls

    spec = create_app().openapi()
    paths = set(spec["paths"])

    by_tag: dict[str, int] = {}
    for ops in spec["paths"].values():
        for op in ops.values():
            tag = (op.get("tags") or ["(未分类)"])[0]
            by_tag[tag] = by_tag.get(tag, 0) + 1

    for tag, minimum in sorted(EXPECTED_MODULES.items()):
        count = by_tag.get(tag, 0)
        report(
            f"模块 {tag} 已注册（{count} 个接口，期望 ≥ {minimum}）",
            count >= minimum,
            "" if count >= minimum else "该模块可能忘了挂路由",
        )

    extra = set(by_tag) - set(EXPECTED_MODULES) - {"meta"}
    report("没有未登记的新模块", not extra, f"多出：{sorted(extra)}" if extra else "")

    total = sum(by_tag.values())
    report(f"接口总数 {total} 条", total >= 30)

    pages = frontend_page_urls("http://127.0.0.1:8000")
    report("启动横幅列出学生端页面", len(pages) == 1, " / ".join(pages))
    return paths


# ---------------------------------------------------------------------------
# 2. 依赖清单一致性
# ---------------------------------------------------------------------------
def check_requirements() -> None:
    """比对 requirements.txt 与 pyproject.toml 的依赖。"""
    print()
    print("=" * 72)
    print("2. 依赖清单一致性（requirements.txt ↔ pyproject.toml）")
    print("=" * 72)

    def names(lines: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in lines:
            text = line.split("#")[0].strip()
            if not text:
                continue
            name = re.split(r"[><=\[]", text)[0].strip().lower()
            result[name] = text
        return result

    pyproject = tomllib.loads((BASE / "pyproject.toml").read_text(encoding="utf-8"))
    declared = names(pyproject["project"]["dependencies"])
    installed = names((BASE / "requirements.txt").read_text(encoding="utf-8").splitlines())

    missing = sorted(set(declared) - set(installed))
    report(
        "requirements.txt 覆盖了 pyproject 的全部依赖",
        not missing,
        f"缺少：{missing}" if missing else "",
    )
    extra = sorted(set(installed) - set(declared))
    report(
        "requirements.txt 没有多余依赖（coverage/docker 例外见注释）",
        set(extra) <= {"coverage", "docker"},
        f"多出：{extra}" if extra else "",
    )
    report(f"两边共有 {len(set(declared) & set(installed))} 个包", len(declared) >= 20)


# ---------------------------------------------------------------------------
# 3. README 里写的接口必须真实存在
# ---------------------------------------------------------------------------
def check_readme_endpoints(paths: set[str]) -> None:
    """把 README 表格里的 `/api/v1/...` 路径抠出来，逐条核对后端确实注册了。"""
    print()
    print("=" * 72)
    print("3. README 里的接口是否真实存在")
    print("=" * 72)

    readme = (BASE / "README.md").read_text(encoding="utf-8")
    # 表格里的路径写成 `GET | /api/v1/xxx`，也可能出现在代码块里的 curl 命令中
    documented = set(re.findall(r"/api/v1/[a-zA-Z0-9_/{}\-]+", readme))
    # 去掉正则误抓到的结尾标点与无关片段
    cleaned = {
        item.rstrip(".,;:、）)") for item in documented
    }

    unknown: list[str] = []
    for path in sorted(cleaned):
        if path in paths:
            continue
        # 允许文档里写的是"前缀"（例如 /api/v1/tutor/history/{id} 在代码里叫 {record_id}）
        normalized = re.sub(r"\{[^}]+\}", "{param}", path)
        if any(
            re.sub(r"\{[^}]+\}", "{param}", registered) == normalized
            for registered in paths
        ):
            continue
        # 也允许写模块前缀（"学生端都在 /api/v1/tutor 下"）：只要它是某个真实路径的前缀
        if any(registered.startswith(path + "/") for registered in paths):
            continue
        unknown.append(path)

    report(
        f"README 提到的 {len(cleaned)} 个接口都真实存在",
        not unknown,
        f"文档里有但后端没有：{unknown}" if unknown else "",
    )

    # 反向：后端每个 URL 前缀（模块）都应当在 README 里出现过。
    # 注意不能用模块**标签**去比对：标签是 `retrieval`，而它的 URL 前缀是
    # `/api/v1/search`，两者不同名；拿标签去 README 里找会永远找不到。
    # 取路径的前 4 段作为模块前缀：/api/v1/tutor/status -> /api/v1/tutor
    # （用 split 时要记住开头那个空串，所以是 [:4] 而不是 [:3]）
    registered_prefixes = {"/".join(path.split("/")[:4]) for path in paths}
    undocumented = sorted(
        prefix
        for prefix in registered_prefixes
        if not any(item.startswith(prefix) for item in cleaned)
    )
    report(
        f"README 覆盖了全部 {len(registered_prefixes)} 个接口前缀",
        not undocumented,
        f"文档没提到：{undocumented}" if undocumented else "",
    )


# ---------------------------------------------------------------------------
# 4. README 章节完整性
# ---------------------------------------------------------------------------
def markdown_slugs(text: str) -> set[str]:
    """把文档里的标题按 GitHub 的规则转成锚点 id。

    GitHub 的规则：转小写 → 去掉标点（保留中英文、数字、下划线、连字符）→ **每个空格**换成连字符。
    自己实现一遍是为了让"目录里的链接点不点得开"这件事能被自动检查，
    否则文档一改标题，目录就悄悄失效了（读者只会觉得这份文档没人维护）。

    注意空格是**一个一个**换成连字符，不是把连续空格压成一个：
    标题"生成测试 → 自动修复"去掉箭头后会留下两个空格，GitHub 给出的锚点是
    `生成测试--自动修复`（两个连字符）。压缩空格会让文档里本来正确的链接被判成坏的
    ——这个坑第一版就踩了。
    """
    slugs: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if not match:
            continue
        title = match.group(1).strip().lower()
        # 去掉 Markdown 行内标记与标点，保留中英文、数字、下划线与连字符
        title = re.sub(r"[`*]", "", title)
        title = re.sub(r"[^\w\u4e00-\u9fff\s-]", "", title)
        slugs.add(re.sub(r"\s", "-", title.strip()))
    return slugs


def check_doc_anchors() -> None:
    """检查两份文档里的目录/交叉引用链接是否都能跳到真实标题。"""
    print()
    print("=" * 72)
    print("4.5 文档内部链接（目录能不能点开）")
    print("=" * 72)

    for name in ("README.md", "使用教程.md"):
        text = (BASE / name).read_text(encoding="utf-8")
        slugs = markdown_slugs(text)
        # 只检查文档内部的锚点链接：](#xxx)
        links = set(re.findall(r"\]\(#([^)]+)\)", text))
        broken = sorted(link for link in links if link not in slugs)
        report(
            f"{name} 的 {len(links)} 个内部链接都能跳到真实标题",
            not broken,
            f"点不开的链接：{broken}" if broken else "",
        )


def check_readme_sections() -> None:
    """检查 README 是否包含需求点名的章节。"""
    print()
    print("=" * 72)
    print("4. README 章节完整性")
    print("=" * 72)

    readme = (BASE / "README.md").read_text(encoding="utf-8")
    for section in EXPECTED_README_SECTIONS:
        report(f"README 含「{section}」章节", section in readme)

    report("README 说明了 .bat 一键启动", "start.bat" in readme and "stop.bat" in readme)
    report("README 说明了接口文档地址", "/docs" in readme)
    report("README 说明了数据存放位置", "app.db" in readme and "data/library" in readme)


# ---------------------------------------------------------------------------
# 5. 前端资源
# ---------------------------------------------------------------------------
def check_frontend() -> None:
    """检查前端文件是否齐全。"""
    print()
    print("=" * 72)
    print("5. 前端页面与静态资源")
    print("=" * 72)

    for relative in EXPECTED_FRONTEND_FILES:
        path = BASE / relative
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        report(f"{relative}（{size} 字节）", exists and size > 100)

    # 学生端页面必须有中文注释（需求要求），且不再引用已删除的课堂展示页
    index_html = (BASE / "frontend" / "index.html").read_text(encoding="utf-8")
    report("学生端页面含中文注释", "<!--" in index_html and "学生端" in index_html)
    report("页面里没有指向已删除文件的死链", "classroom.html" not in index_html)


# ---------------------------------------------------------------------------
# 6. .bat 脚本
# ---------------------------------------------------------------------------
def check_bat() -> None:
    """检查 .bat 是否存在、编码是否正确、是否打印了正确的页面地址。"""
    print()
    print("=" * 72)
    print("6. 一键启动脚本（.bat）")
    print("=" * 72)

    for name in ("start.bat", "stop.bat"):
        path = BASE / name
        if not path.is_file():
            report(f"{name} 存在", False)
            continue
        raw = path.read_bytes()
        text = raw.decode("gbk", errors="replace")
        report(f"{name} 存在（{len(raw)} 字节）", True)
        # Windows 批处理必须是 CRLF：裸 LF 会让 cmd 把 REM 前缀丢掉并刷一屏报错
        bare_lf = raw.replace(b"\r\n", b"").count(b"\n")
        report(f"{name} 使用 CRLF 换行（裸 LF {bare_lf} 个）", bare_lf == 0)
        report(f"{name} 用 GBK 编码且含中文提示", "正在" in text or "服务" in text)

    start = (BASE / "start.bat").read_bytes().decode("gbk", errors="replace")
    report("start.bat 打印学生端地址", "学生界面" in start)
    report("start.bat 打印接口文档地址", "/docs" in start)

    # 编码规则：chcp 之前**不能出现中文**。
    # 注意不是"chcp 必须是第一行"——`@echo off` 这类纯 ASCII 命令排在它前面没问题，
    # 真正会让 cmd 用旧代码页解码中文字节从而乱码的，是 chcp 之前出现中文。
    lines = start.splitlines()
    chcp_index = next(
        (index for index, row in enumerate(lines) if row.strip().lower().startswith("chcp")),
        -1,
    )
    report("start.bat 里有 chcp", chcp_index >= 0)
    if chcp_index >= 0:
        offenders = [
            index + 1
            for index, row in enumerate(lines[:chcp_index])
            if any("\u4e00" <= char <= "\u9fff" for char in row)
        ]
        report(
            f"chcp（第 {chcp_index + 1} 行）之前没有中文",
            not offenders,
            f"这些行有中文：{offenders}" if offenders else "",
        )


# ---------------------------------------------------------------------------
# 7. 数据库表
# ---------------------------------------------------------------------------
def check_database(tmp_dir: Path) -> None:
    """确认所有记录表都能建出来。"""
    print()
    print("=" * 72)
    print("7. 数据库表结构")
    print("=" * 72)

    import asyncio
    import sqlite3

    db_path = tmp_dir / "integration.db"

    async def build() -> None:
        from app.core.config import get_settings
        from app.core.database import dispose_engine, init_db

        get_settings.cache_clear()
        await init_db()
        await dispose_engine()

    import os

    os.environ["DATABASE__SQLITE_PATH"] = str(db_path)
    asyncio.run(build())

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    missing = sorted(EXPECTED_TABLES - tables)
    report(
        f"{len(EXPECTED_TABLES)} 张业务表都建出来了",
        not missing,
        f"缺少：{missing}" if missing else "",
    )
    print(f"       实际表：{sorted(tables)}")


# ---------------------------------------------------------------------------
# 8. 核心实现文件
# ---------------------------------------------------------------------------
def check_service_files() -> None:
    """确认各模块的实现文件都在（漏文件同样是"静默失败"）。"""
    print()
    print("=" * 72)
    print("8. 核心实现文件")
    print("=" * 72)

    for relative in EXPECTED_SERVICE_FILES:
        if relative.endswith("classifier_placeholder"):
            relative = "app/services/file_classifier.py"
        path = BASE / relative
        ok = path.is_file()
        report(relative, ok)


def main() -> int:
    import tempfile

    print("=" * 72)
    print("整合自检：确认所有模块、依赖、文档、脚本都真的接上了")
    print("=" * 72)
    print()

    paths = check_routes()
    check_requirements()
    check_readme_endpoints(paths)
    check_readme_sections()
    check_doc_anchors()
    check_frontend()
    check_bat()
    # ignore_cleanup_errors：Windows 上 SQLite 连接刚释放时文件还可能被占用，
    # 临时目录删不掉会抛 PermissionError，那属于清理问题而不是自检失败。
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        check_database(Path(tmp))
    check_service_files()

    print()
    print("=" * 72)
    passed = sum(1 for _, ok in checks if ok)
    print(f"整合自检结果：{passed}/{len(checks)} 项通过")
    print("=" * 72)
    failed = [label for label, ok in checks if not ok]
    for label in failed:
        print(f"  FAIL  {label}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
