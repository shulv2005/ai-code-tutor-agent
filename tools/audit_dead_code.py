"""死代码审计：找出没人引用的模块、函数、CSS 类名与可清理的产物。

用法（项目根目录下执行，只读、不会删任何东西）：
    python tools/audit_dead_code.py

审计四件事：
  1. app/ 下有没有从未被任何文件引用的 Python 模块
  2. app/ 下有没有定义了却没人调用的顶层函数 / 类
  3. frontend/css 里有哪个类名在 HTML/JS 里根本没用过
  4. 项目根目录有没有该清理的缓存 / 构建产物 / 临时文件

为什么要专门写这个脚本：
  功能一多就容易留下"当时有用、后来没人用"的代码。靠肉眼看是看不住的，
  用脚本每次改完跑一遍，能立刻指出"这块可以删了"，比事后翻代码可靠得多。
  这次清理就是先跑它、再逐条确认、最后又跑一遍确认归零。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# 不参与"是否被引用"统计的目录：第三方、缓存、下载的模型权重
SKIP_DIRS = {
    ".venv", "data", "__pycache__", ".pytest_cache", ".ruff_cache",
    "node_modules", ".git", "opensource_collab_agent.egg-info",
}
SOURCE_SUFFIXES = {
    ".py", ".js", ".html", ".css", ".md", ".bat", ".txt", ".toml", ".json",
    ".example", ".yaml", ".yml",
}

# 这些名字由框架按约定调用（或本身就是入口），只出现一次是正常的
FRAMEWORK_CALLED = {
    "lifespan", "main", "app", "settings", "get_settings", "create_app",
    "setup_logging",
}

# 由 JS 模板字符串拼出来的 CSS 类名：正则扫不到，但确实在用。
# 例：app.js 里的 `issue--${issue.severity}` 会生成 issue--error / issue--warning / issue--info
DYNAMIC_CSS_CLASSES = {
    "issue--error", "issue--warning", "issue--info",
}


def iter_files(suffixes: set[str] | None = None):
    """遍历项目里的文本文件，跳过缓存与数据目录。"""
    for path in BASE.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(BASE).parts):
            continue
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        yield path


def module_name(path: Path) -> str:
    """app/services/repo/scanner.py -> app.services.repo.scanner"""
    return ".".join(path.relative_to(BASE).with_suffix("").parts)


def read_text(path: Path) -> str:
    """读文本；编码异常的文件直接当空文件处理，不让审计本身崩掉。"""
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def audit_modules(corpus: str) -> list[str]:
    """1. 从未被 import 的模块。"""
    print("=" * 72)
    print("1. app/ 下从未被引用的模块")
    print("=" * 72)

    orphans: list[str] = []
    total = 0
    for path in sorted((BASE / "app").rglob("*.py")):
        total += 1
        name = module_name(path)
        # 包自身的 __init__.py 与入口 main.py 不作要求
        if path.name in {"__init__.py", "main.py"}:
            continue
        short = name.rsplit(".", 1)[-1]
        # 只要出现过完整路径或模块短名，就认为被引用过
        if name in corpus or short in corpus:
            continue
        orphans.append(name)

    print(f"  扫描模块 {total} 个，疑似无引用 {len(orphans)} 个")
    for name in orphans:
        print(f"    · {name}")
    print()
    return orphans


def audit_symbols(corpus: str) -> list[str]:
    """2. 定义了但全项目只出现一次的顶层函数 / 类（= 没人调用）。"""
    print("=" * 72)
    print("2. app/ 下定义了却没人调用的顶层函数 / 类")
    print("=" * 72)

    suspects: list[str] = []
    checked = 0
    for path in sorted((BASE / "app").rglob("*.py")):
        try:
            tree = ast.parse(read_text(path))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            checked += 1
            if node.name.startswith("__") or node.name in FRAMEWORK_CALLED:
                continue
            # 带装饰器的（路由、validator、fixture…）由框架调用，不算死代码
            if node.decorator_list:
                continue
            if len(re.findall(rf"\b{re.escape(node.name)}\b", corpus)) <= 1:
                suspects.append(f"{path.relative_to(BASE).as_posix()}::{node.name}")

    print(f"  检查顶层定义 {checked} 个，疑似无人调用 {len(suspects)} 个")
    for item in suspects:
        print(f"    · {item}")
    print()
    return suspects


def audit_css_classes() -> list[str]:
    """3. CSS 里定义但 HTML/JS 没用到的类名。"""
    print("=" * 72)
    print("3. CSS 里定义但 HTML/JS 没用到的类名")
    print("=" * 72)

    frontend = BASE / "frontend"
    used_text = "\n".join(
        read_text(p) for p in frontend.rglob("*")
        if p.is_file() and p.suffix.lower() in {".html", ".js"}
    )

    unused: list[str] = []
    defined = 0
    for css_file in sorted((frontend / "css").glob("*.css")):
        # 先去掉注释再找选择器，避免注释里提到的类名被算成"定义"
        css = re.sub(r"/\*.*?\*/", "", read_text(css_file), flags=re.DOTALL)
        names = set(re.findall(r"\.([a-zA-Z][a-zA-Z0-9_-]*)", css))
        defined += len(names)
        for name in sorted(names):
            if name in DYNAMIC_CSS_CLASSES:
                continue
            # 独立词匹配：.btn 不能因为 .btn--mini 出现过就算被用到
            if not re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", used_text):
                unused.append(f"{css_file.name}: .{name}")

    print(f"  CSS 定义类名 {defined} 个，疑似未使用 {len(unused)} 个")
    for item in unused:
        print(f"    · {item}")
    print()
    return unused


def audit_artifacts() -> list[str]:
    """4. 顶层目录体检：缓存 / 构建产物 / 临时文件。"""
    print("=" * 72)
    print("4. 顶层目录体检（缓存 / 构建产物 / 临时文件）")
    print("=" * 72)

    allowed_dirs = {
        "app", "tests", "tools", "frontend", "data", "examples", "docker", ".venv",
    }
    junk: list[str] = []
    for entry in sorted(BASE.iterdir()):
        name = entry.name
        if name in {".pytest_cache", ".ruff_cache", "opensource_collab_agent.egg-info"}:
            junk.append(f"{name}/  （可重新生成的缓存/构建产物）")
        elif entry.is_file() and (name.startswith("_") or name.endswith(".tmp")):
            junk.append(f"{name}  （临时文件）")
        elif entry.is_dir() and name not in allowed_dirs:
            junk.append(f"{name}/  （不在项目结构里的目录）")

    for item in junk:
        print(f"  · {item}")
    if not junk:
        print("  没有发现需要清理的产物 ✓")
    print()
    return junk


def main() -> int:
    corpus = "\n".join(read_text(p) for p in iter_files(SOURCE_SUFFIXES | {".env"}))

    print()
    modules = audit_modules(corpus)
    symbols = audit_symbols(corpus)
    classes = audit_css_classes()
    artifacts = audit_artifacts()

    pending = len(modules) + len(symbols) + len(classes) + len(artifacts)
    print("=" * 72)
    if pending:
        print(f"审计结果：有 {pending} 处待确认（脚本只报告，不会自动删除）")
    else:
        print("审计结果：没有发现死代码或多余产物 ✓")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
