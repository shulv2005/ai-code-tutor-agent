"""整合测试：确认所有模块、依赖、文档、脚本都真的接上了。

与 `tools/verify_integration.py` 的分工：
那个脚本给人看（逐项打印，适合上课前跑一遍）；
这里把同样的关键不变量固化进 pytest，让"漏挂一个模块"在 CI 里就失败。

这些用例守的都是**不会报错、只会让人在课上卡住**的问题：
模块忘了挂（访问时 404）、依赖清单不一致（换台机器装不起来）、
README 写的接口不存在（学生照着敲调不通）。
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from app.core.config import PROJECT_ROOT
from app.main import FRONTEND_PAGES, create_app, frontend_page_urls

# 期望的模块标签 -> 最少接口数（只写下限，以后加接口不用改）
EXPECTED_MODULES = {
    "health": 2,
    "repositories": 7,
    "retrieval": 1,
    "agent": 3,
    "sandbox": 2,
    "tutor": 8,
    "library": 3,
    "files": 2,
    "check": 1,
    "fix": 3,
    "comment": 3,
    # 多模型与网页填 Key：3 个会话接口 + 1 个模型清单接口
    "auth": 3,
    "models": 1,
}

# 每个功能模块的落库表
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


@pytest.fixture(scope="module")
def spec() -> dict:
    """整份 OpenAPI schema（所有模块加载一次）。"""
    return create_app().openapi()


def _module_counts(spec: dict) -> dict[str, int]:
    """统计每个模块标签下的接口数。"""
    counts: dict[str, int] = {}
    for operations in spec["paths"].values():
        for operation in operations.values():
            tag = (operation.get("tags") or ["(未分类)"])[0]
            counts[tag] = counts.get(tag, 0) + 1
    return counts


# ===========================================================================
# 1. 模块整合
# ===========================================================================
def test_all_modules_are_registered(spec: dict) -> None:
    """13 个模块必须全部挂上——少一个不会报错，只会在访问时 404。"""
    counts = _module_counts(spec)
    missing = {
        tag: counts.get(tag, 0)
        for tag, minimum in EXPECTED_MODULES.items()
        if counts.get(tag, 0) < minimum
    }
    assert not missing, f"这些模块的接口数不足（可能忘了挂路由）: {missing}"


def test_no_unknown_module_appears(spec: dict) -> None:
    """出现新模块时要同步更新本测试，避免模块被无声改动。"""
    extra = set(_module_counts(spec)) - set(EXPECTED_MODULES)
    assert not extra, f"出现未登记的模块标签: {sorted(extra)}"


def test_router_mounts_every_endpoint_module() -> None:
    """路由聚合文件必须把每个 endpoints 模块都 include 进来。"""
    router_source = (PROJECT_ROOT / "app" / "api" / "v1" / "router.py").read_text(
        encoding="utf-8"
    )
    endpoint_files = sorted(
        path.stem
        for path in (PROJECT_ROOT / "app" / "api" / "v1" / "endpoints").glob("*.py")
        if path.stem != "__init__"
    )
    assert endpoint_files, "endpoints 目录下应当有模块文件"

    unmounted = [name for name in endpoint_files if f"{name}.router" not in router_source]
    assert not unmounted, f"这些模块写了但没挂到路由上: {unmounted}"


def test_main_is_a_thin_assembly_layer() -> None:
    """main.py 只负责组装：业务模块一律通过 api_router 挂载。"""
    source = (PROJECT_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    assert "include_router(api_router" in source
    # 不允许在 main.py 里直接挂业务路由（那会让"模块在哪挂的"变得难找）
    assert "from app.api.v1.endpoints" not in source


def test_startup_inventory_helper_reports_modules() -> None:
    """启动时的路由清单函数要能报出所有模块（整合是否完整的自查表）。"""
    from app.main import log_route_inventory

    inventory = log_route_inventory(create_app())
    assert set(EXPECTED_MODULES) <= set(inventory)
    assert sum(inventory.values()) >= 30


def test_frontend_pages_are_declared() -> None:
    """两个页面都要登记在 main.py 的清单里。"""
    names = [name for name, _ in FRONTEND_PAGES]
    assert names == ["index.html"]
    urls = frontend_page_urls("http://127.0.0.1:8000")
    assert urls[0].startswith("http://127.0.0.1:8000/ui/index.html")


# ===========================================================================
# 2. 依赖清单
# ===========================================================================
def test_requirements_covers_pyproject_dependencies() -> None:
    """requirements.txt 少了包，换台机器就会装不起来。"""
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        re.split(r"[><=\[]", item)[0].strip().lower()
        for item in pyproject["project"]["dependencies"]
    }
    requirements = {
        re.split(r"[><=\[]", line.split("#")[0].strip())[0].strip().lower()
        for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.split("#")[0].strip()
    }

    missing = sorted(declared - requirements)
    assert not missing, f"requirements.txt 缺少: {missing}"
    # coverage/docker 在 requirements 里是运行期依赖（本地沙箱后端需要），
    # 不算多余；除此之外两边应当完全一致
    assert (requirements - declared) <= {"coverage", "docker"}


def test_requirements_mentions_key_packages() -> None:
    """几个"缺了会直接报错"的包必须在清单里写清楚。"""
    text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for package in ("fastapi", "python-multipart", "tree-sitter-c", "tree-sitter-java", "docker"):
        assert package in text, f"requirements.txt 缺少 {package}"


# ===========================================================================
# 3. README 与实现的对应关系
# ===========================================================================
def test_readme_documented_endpoints_exist(spec: dict) -> None:
    """README 里列出的每个接口路径都必须真实存在（学生照文档敲要能通）。

    允许文档写「模块前缀」（例如"学生端都在 `/api/v1/tutor` 下"）：
    只要它是某个已注册路径的前缀就算数。否则为了让这句正常的中文表述通过，
    文档就得刻意避开模块名，反而把说明写得更别扭。
    """
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    documented = {
        item.rstrip(".,;:、）)")
        for item in re.findall(r"/api/v1/[a-zA-Z0-9_/{}\-]+", readme)
    }
    registered = set(spec["paths"])
    normalized_registered = {
        re.sub(r"\{[^}]+\}", "{param}", path) for path in registered
    }

    unknown = sorted(
        path
        for path in documented
        if path not in registered
        and re.sub(r"\{[^}]+\}", "{param}", path) not in normalized_registered
        and not any(item.startswith(path + "/") for item in registered)
    )
    assert not unknown, f"README 里有后端不存在的接口: {unknown}"


def test_readme_covers_every_module_prefix(spec: dict) -> None:
    """反向检查：后端每个 URL 前缀都要在 README 里出现过。"""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    # 前 4 段才是模块前缀：/api/v1/tutor/status -> /api/v1/tutor
    prefixes = {"/".join(path.split("/")[:4]) for path in spec["paths"]}
    undocumented = sorted(prefix for prefix in prefixes if prefix not in readme)
    assert not undocumented, f"README 没提到这些模块: {undocumented}"


@pytest.mark.parametrize(
    "section",
    ["项目功能", "一键启动", "如何使用 HTML 界面", "如何添加学生代码", "AI 功能的原理"],
)
def test_readme_has_required_sections(section: str) -> None:
    """需求点名的章节一个都不能少。"""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert section in readme, f"README 缺少「{section}」章节"


def test_readme_explains_three_ai_features() -> None:
    """检测 / 改错 / 注释三个功能的原理都要有说明（含本地+AI 两阶段）。"""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for feature in ("AI 自动检测", "AI 代码改错", "AI 注释生成"):
        assert feature in readme, f"README 没有说明 {feature}"
    assert "tree-sitter" in readme.lower() or "Tree-sitter" in readme
    assert "本地复检" in readme or "本地复核" in readme


# ===========================================================================
# 4. 静态资源与脚本
# ===========================================================================
@pytest.mark.parametrize(
    "relative",
    [
        "frontend/index.html",
            "frontend/css/style.css",
            "frontend/js/app.js",
            "frontend/vendor/highlight.min.js",
        "frontend/vendor/github.min.css",
        "start.bat",
        "stop.bat",
        "requirements.txt",
        ".env.example",
    ],
)
def test_integration_artifacts_exist(relative: str) -> None:
    """整合后的产物必须都在仓库里。"""
    path = PROJECT_ROOT / relative
    assert path.is_file(), f"{relative} 不存在"
    assert path.stat().st_size > 100, f"{relative} 内容过少，可能是空文件"


def test_bat_files_are_windows_ready() -> None:
    """批处理必须是 GBK + CRLF，且 chcp 之前不能有中文。

    裸 LF 会让 cmd.exe 丢掉 REM 前缀并刷一屏报错；
    chcp 之前出现中文会被按旧代码页解码成乱码。这两条都踩过坑。
    """
    for name in ("start.bat", "stop.bat"):
        raw = (PROJECT_ROOT / name).read_bytes()
        assert raw.replace(b"\r\n", b"").count(b"\n") == 0, f"{name} 里有裸 LF"

        text = raw.decode("gbk", errors="replace")
        lines = text.splitlines()
        chcp_index = next(
            (index for index, row in enumerate(lines) if row.strip().lower().startswith("chcp")),
            -1,
        )
        assert chcp_index >= 0, f"{name} 缺少 chcp"
        offenders = [
            index + 1
            for index, row in enumerate(lines[:chcp_index])
            if any("\u4e00" <= char <= "\u9fff" for char in row)
        ]
        assert not offenders, f"{name} 的 chcp 之前出现了中文（第 {offenders} 行）"


def test_start_bat_advertises_both_pages() -> None:
    """start.bat 的启动横幅要告诉用户两个页面在哪。"""
    text = (PROJECT_ROOT / "start.bat").read_bytes().decode("gbk", errors="replace")
    assert "学生界面" in text
    assert "/docs" in text


# ===========================================================================
# 4.5 前端显隐控制（回归：hidden 属性被 CSS 覆盖导致提示消不掉）
# ===========================================================================
@pytest.mark.parametrize(
    "stylesheet",
    ["frontend/css/style.css"],
)
def test_stylesheet_keeps_hidden_fallback_rule(stylesheet: str) -> None:
    """样式表必须保留 `[hidden] { display: none !important }`。

    回归用例（用户实际反馈过的问题）：页面靠 `el.hidden = true/false` 控制显隐，
    但 `hidden` 的效果来自浏览器默认样式的 `[hidden] { display: none }`，
    而**作者样式里的任何 display 声明都会覆盖它**（与选择器权重无关）。
    缺了这条兜底规则，`.empty { display: grid }` 就会让
    「还没有代码 / 从左边选择文件，或粘贴一段代码」一直压在代码上面，
    `.result { display: flex }` 会让三块结果面板同时堆着显示。

    这里同时校验 `!important`：没有它，权重更高的选择器仍可能翻盘。
    """
    css = (PROJECT_ROOT / stylesheet).read_text(encoding="utf-8")
    rule = re.search(r"\[hidden\]\s*\{[^}]*display\s*:\s*none\s*!important", css)
    assert rule, f"{stylesheet} 缺少 [hidden] {{ display: none !important }} 兜底规则"


@pytest.mark.parametrize(
    ("html_path", "js_path", "css_path"),
    [
        # 学生端是当前唯一的前端页面；以后新增页面时在这里补一行即可
        ("frontend/index.html", "frontend/js/app.js", "frontend/css/style.css"),
    ],
)
def test_no_hidden_element_is_defeated_by_css(
    html_path: str, js_path: str, css_path: str
) -> None:
    """凡是"JS 用 .hidden 控制、自身类名又带 display 声明"的元素都要被兜底规则覆盖。

    这条检查的价值在于：以后有人给某个卡片加了 `display: flex`，
    这个用例会立刻指出"它依赖那条兜底规则"，而不是等到课堂上发现提示消不掉。
    """
    html = (PROJECT_ROOT / html_path).read_text(encoding="utf-8")
    js = (PROJECT_ROOT / js_path).read_text(encoding="utf-8")
    css = (PROJECT_ROOT / css_path).read_text(encoding="utf-8")

    toggled = set(re.findall(r"(\w+)\.hidden\s*=", js))
    el_map = dict(re.findall(r"(\w+):\s*document\.getElementById\('([^']+)'\)", js))

    classes: dict[str, list[str]] = {}
    for match in re.finditer(r"<[^>]*id=\"([^\"]+)\"[^>]*>", html):
        found = re.search(r'class="([^"]+)"', match.group(0))
        classes[match.group(1)] = found.group(1).split() if found else []

    risky: list[str] = []
    for var in sorted(toggled):
        element_id = el_map.get(var, var)
        for class_name in classes.get(element_id, []):
            for rule in re.finditer(
                r"\." + re.escape(class_name) + r"\s*(?:,[^{]*)?\{([^}]*)\}", css
            ):
                if "display" in rule.group(1):
                    risky.append(f"{element_id} (.{class_name})")

    # 有风险元素时，兜底规则必须在（否则这些元素的 hidden 全是失效的）
    if risky:
        assert re.search(
            r"\[hidden\]\s*\{[^}]*display\s*:\s*none\s*!important", css
        ), f"{css_path} 里这些元素的 hidden 会被覆盖，但缺少兜底规则: {sorted(set(risky))}"

    # 顺带保证检查本身是有效的：这两个页面确实存在这类元素
    assert risky, f"{html_path} 里应当有靠 .hidden 控制且带 display 的元素（检查逻辑可能失效了）"


# ===========================================================================
# 5. 数据库表
# ===========================================================================
def test_all_tables_are_created(sqlite_path: Path) -> None:
    """所有功能模块的落库表都要能建出来（模型忘了在 __init__ 导出就不会建）。"""
    import asyncio
    import sqlite3

    from app.core.database import dispose_engine, init_db

    async def build() -> None:
        await init_db()
        await dispose_engine()

    asyncio.run(build())

    with sqlite3.connect(sqlite_path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    missing = sorted(EXPECTED_TABLES - tables)
    assert not missing, f"缺少数据表: {missing}（检查 app/models/__init__.py 是否导出）"


def test_every_model_is_exported() -> None:
    """models 包要把每个模型模块都导出，否则 create_all 不会建表。"""
    models_init = (PROJECT_ROOT / "app" / "models" / "__init__.py").read_text(encoding="utf-8")
    model_modules = sorted(
        path.stem
        for path in (PROJECT_ROOT / "app" / "models").glob("*.py")
        if path.stem not in {"__init__", "repository", "code"}
    )
    for module in model_modules:
        assert f"app.models.{module}" in models_init, f"{module} 没有在 models/__init__.py 里导出"
