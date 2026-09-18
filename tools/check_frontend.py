"""检查前端文件的完整性：JS 语法、DOM id 对应关系、CSS 类名、后端接口是否都对得上。

覆盖学生端页面：
    frontend/index.html  + js/app.js  + css/style.css

为什么要做成"按页面配置"而不是写死一份：
新增第二个页面时如果只复制粘贴一套检查逻辑，很容易漏掉新页面的资源；
这里把页面清单列在 PAGES 里，新增页面只要加一行。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# 每个页面：HTML、JS、CSS，以及该页调用的接口前缀
PAGES = [
    {
        "name": "学生端",
        "html": BASE / "frontend" / "index.html",
        "js": BASE / "frontend" / "js" / "app.js",
        "css": BASE / "frontend" / "css" / "style.css",
        "bases": ("/api/v1/tutor", "/api/v1/library", "/api/v1/files"),
    },
]

failures: list[str] = []


def check_syntax(page: dict) -> None:
    """1. JS 语法检查（node --check）。"""
    print("=" * 72)
    print(f"1. JS 语法检查（{page['name']}：{page['js'].name}）")
    print("=" * 72)
    result = subprocess.run(
        ["node", "--check", str(page["js"])],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode == 0:
        print("  OK  语法正确")
    else:
        failures.append(f"{page['js'].name} 语法错误")
        print("  FAIL 语法错误:")
        print(result.stderr[:800])


def check_dom_ids(page: dict, html: str, js: str) -> None:
    """2. DOM id 对应关系：JS 取的 id 必须都在 HTML 里。"""
    print()
    print("=" * 72)
    print(f"2. DOM id 对应关系（{page['name']}）")
    print("=" * 72)
    html_ids = set(re.findall(r'id="([^"]+)"', html))
    js_ids = set(re.findall(r"getElementById\('([^']+)'\)", js))
    missing = js_ids - html_ids
    print(f"  HTML 定义 id : {len(html_ids)} 个")
    print(f"  JS 引用 id   : {len(js_ids)} 个")
    print(f"  JS 引用但 HTML 没有 : {sorted(missing) if missing else '无 ✓'}")
    if missing:
        failures.append(f"{page['name']} 有 {len(missing)} 个 id 对不上")


def check_css_classes(page: dict, html: str, js: str, css: str) -> None:
    """3. CSS 类名对应关系：页面用到的类名应当在 CSS 里有定义。"""
    print()
    print("=" * 72)
    print(f"3. CSS 类名对应关系（{page['name']}）")
    print("=" * 72)
    html_classes: set[str] = set()
    for attr in re.findall(r'class="([^"]+)"', html):
        html_classes.update(attr.split())
    js_classes = set(re.findall(r"class=[\"']([a-z0-9_\- ]+)[\"']", js))
    js_classes = {c for item in js_classes for c in item.split()}
    css_classes = set(re.findall(r"\.([a-zA-Z][a-zA-Z0-9_-]*)", css))
    used = html_classes | js_classes
    undefined = sorted(c for c in used if c not in css_classes)
    print(f"  CSS 定义类名 : {len(css_classes)} 个")
    print(f"  页面用到类名 : {len(used)} 个")
    print(f"  用到但 CSS 未定义 : {undefined if undefined else '无 ✓'}")
    if undefined:
        failures.append(f"{page['name']} 有 {len(undefined)} 个类名未定义")


def check_hidden_toggle(page: dict, html: str, js: str, css: str) -> None:
    """3.5 显隐控制检查：JS 用 .hidden 控制的元素，CSS 不能把 hidden 覆盖掉。

    这一条是踩过坑之后加的：
    `hidden` 属性的效果来自浏览器默认样式表的 `[hidden] { display: none }`，
    而**作者样式表里任何 display 声明都会覆盖它**（与选择器权重无关）。
    后果是 `el.codeEmpty.hidden = true` 完全不起作用——
    "还没有代码 / 从左边选择文件，或粘贴一段代码"一直压在代码上面，
    三个结果面板也会同时堆着显示。

    检查两件事：
      ① CSS 里必须有 `[hidden] { display: none }` 兜底（缺了就算失败）；
      ② 顺便列出"被 .hidden 控制、自身类名又带 display 声明"的元素，
         让开发者知道哪些地方依赖那条兜底规则。
    """
    print()
    print("=" * 72)
    print(f"3.5 显隐控制（{page['name']}：hidden 属性是否真的生效）")
    print("=" * 72)

    has_rule = bool(re.search(r"\[hidden\]\s*\{[^}]*display\s*:\s*none", css))
    print(f"  CSS 有 [hidden] 兜底规则 : {'是 ✓' if has_rule else '否 ✗'}")
    if not has_rule:
        failures.append(
            f"{page['name']} 缺少 [hidden] {{ display: none }} 兜底规则，"
            "JS 里 el.hidden = true 会被自定义的 display 覆盖"
        )

    toggled = set(re.findall(r"(\w+)\.hidden\s*=", js))
    el_map = dict(re.findall(r"(\w+):\s*document\.getElementById\('([^']+)'\)", js))

    classes: dict[str, list[str]] = {}
    for match in re.finditer(r"<[^>]*id=\"([^\"]+)\"[^>]*>", html):
        found = re.search(r'class="([^"]+)"', match.group(0))
        classes[match.group(1)] = found.group(1).split() if found else []

    risky: set[str] = set()
    for var in sorted(toggled):
        element_id = el_map.get(var, var)
        for class_name in classes.get(element_id, []):
            for rule in re.finditer(
                r"\." + re.escape(class_name) + r"\s*(?:,[^{]*)?\{([^}]*)\}", css
            ):
                if "display" in rule.group(1):
                    risky.add(f"{element_id} (.{class_name})")

    print(f"  被 .hidden 控制的元素 : {len(toggled)} 个")
    print(f"  其中类名带 display 声明的 : {len(risky)} 个"
          f"（{'已由兜底规则覆盖 ✓' if has_rule else '会被覆盖 ✗'}）")
    for item in sorted(risky):
        print(f"      · {item}")


def read_base(js: str, name: str) -> str:
    """从 JS 里读出接口前缀常量（避免两边写死后悄悄对不上）。"""
    found = re.search(rf"""{name}\s*=\s*['"]([^'"]+)['"]""", js)
    return found.group(1) if found else ""


def iter_api_calls(js: str, default_base: str) -> list[tuple[str, str, str]]:
    """逐个找出 api(...) 调用，返回 (路径字面量, 该调用使用的前缀, HTTP 方法)。

    三种写法都要认：单引号、双引号、模板字符串（带 ${变量}）；
    前缀有两种来源，都要能识别出来：
      1. 调用处直接写字符串：api('/x', {}, '/api/v1/other')
      2. 引用页面里的常量：  api('/x', {}, OTHER_BASE)   ← 学生端就是这么写的
    第 2 种不能只看片段里有没有 "/api/v1" 字样，得先把常量表解出来。

    HTTP 方法也要认：同一个路径可能同时有 GET / PUT / DELETE
    （项目库的 /file 就是三个方法），只比路径的话
    "把 PUT 写成 POST" 这类错误会被放过，所以方法必须一起校验。
    """
    # 先把 JS 里所有 `名字 = '/api/v1/...'` 的常量收集起来
    constants = dict(re.findall(r"""([A-Z_][A-Z0-9_]*)\s*=\s*['"](/api/v1[^'"]*)['"]""", js))

    calls: list[tuple[str, str, str]] = []
    for match in re.finditer(r"""api\(\s*[`'"]([^`'"]+)[`'"]""", js):
        # 取"本次调用"的参数片段：到该调用结尾的 `);` 为止。
        # 用 `);` 而不是第一个 `)`，是因为参数里还会有 params.toString() 这种嵌套括号。
        end = js.find(");", match.end())
        segment = js[match.end() : end if end != -1 else match.end() + 200]

        base = default_base
        # ① 调用处直接写的字符串前缀
        for candidate in re.findall(r"['\"](/api/v1/[^'\"]+)['\"]", segment):
            base = candidate
        # ② 引用了常量（写在最后，因此优先级更高：常量就是这一页真实用的前缀）
        for name, value in constants.items():
            if re.search(rf"\b{name}\b", segment):
                base = value

        # 方法：片段里写了 method: 'PUT' 就用它，没写就是 fetch 默认的 GET
        method_match = re.search(r"""method\s*:\s*['"](\w+)['"]""", segment)
        method = method_match.group(1).upper() if method_match else "GET"
        calls.append((match.group(1), base, method))
    return calls


def check_api_paths(page: dict, js: str, operations: dict[str, set[str]]) -> None:
    """4. 前端调用的接口必须都在后端注册过（路径 + 方法都要对得上）。"""
    print()
    print("=" * 72)
    print(f"4. 前端调用的接口 vs 后端已注册接口（{page['name']}）")
    print("=" * 72)

    registered = set(operations)
    api_base = read_base(js, "API_BASE") or page["bases"][0]
    print(f"  前端 API_BASE = {api_base}")

    def normalize(call: str, base: str) -> str:
        """把前端写法归一成后端注册的路径形式（去查询串、占位符统一、补前缀）。"""
        path = call.split("?")[0]
        path = re.sub(r"\$\{[^}]+\}", "{}", path)
        if base and not path.startswith(base):
            path = base + path
        return path

    def match_registered(front_path: str, base: str) -> str | None:
        """按「段数相同 + 固定段一致」的规则，找后端对应的注册路由。"""
        front_parts = normalize(front_path, base).strip("/").split("/")
        for candidate in sorted(registered):
            parts = candidate.strip("/").split("/")
            if len(parts) != len(front_parts):
                continue
            ok = all(
                front == reg or (front == "{}" and reg.startswith("{") and reg.endswith("}"))
                for front, reg in zip(front_parts, parts, strict=True)
            )
            if ok:
                return candidate
        return None

    matched: set[tuple[str, str]] = set()
    local_failures: list[str] = []
    for call, base, method in sorted(set(iter_api_calls(js, api_base))):
        target = match_registered(call, base)
        if target and method in operations[target]:
            matched.add((target, method))
            print(f"  OK  {method:6} api('{call}')  ->  {target}")
        elif target:
            # 路径对、方法错：这类错误最隐蔽，浏览器只会回 405，页面表现为"接口不通"
            local_failures.append(f"{method} {call}")
            print(f"  FAIL  {method:6} api('{call}')  后端 {target} 没有 {method} 方法"
                  f"（支持：{'/'.join(sorted(operations[target])).upper()}）")
        else:
            local_failures.append(f"{method} {call}")
            print(f"  FAIL  {method:6} api('{call}')  后端没有这个接口（前缀 {base}）")

    if local_failures:
        failures.append(f"{page['name']} 有 {len(local_failures)} 个接口调用对不上后端")
        print(f"\n  结论：有 {len(local_failures)} 个前端调用对不上后端接口 ✗")
    else:
        print("\n  结论：本页调用的接口全部存在 ✓")

    # 该页关心的模块下，后端有哪些操作没被这个页面用到
    for base in page["bases"]:
        paths = sorted(p for p in registered if p.startswith(base + "/"))
        total = sum(len(operations[path]) for path in paths)
        unused = [
            f"{method.upper()} {path}"
            for path in paths
            for method in sorted(operations[path])
            if (path, method) not in matched
        ]
        print(f"  {base} 下共 {total} 个接口（{len(paths)} 条路径），本页用了 {total - len(unused)} 个")
        for item in unused:
            print(f"    （本页未调用）{item}")


def check_js_behavior() -> None:
    """5. 前端行为测试：在 Node 里用 DOM 桩真跑一遍 app.js。

    为什么要有这一步：静态检查只能证明"代码里有这些关键字"，
    证明不了"拖动真的会被最小宽度夹住""1000 行真的分批高亮"。
    浏览器自动化在本环境不可用，于是用 `check_frontend_logic.mjs`
    把 app.js 跑起来、直接调函数、模拟 pointer 事件，断言真实行为。

    覆盖：占位提示显隐（需求一）、分隔条拖动与最小宽度（需求三）、
    大文件分批高亮（需求四）。
    """
    print()
    print("=" * 72)
    print("5. 前端行为测试（Node + DOM 桩，真跑 app.js）")
    print("=" * 72)

    script = BASE / "tools" / "check_frontend_logic.mjs"
    if not script.is_file():
        failures.append("缺少 tools/check_frontend_logic.mjs（前端行为测试）")
        print("  FAIL 找不到行为测试脚本")
        return

    result = subprocess.run(
        ["node", str(script), str(BASE)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (result.stdout or "") + (result.stderr or "")

    # 只回显各需求小节的标题与失败项，避免几十行 OK 把屏幕刷满
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(("需求", "前端行为测试")) or stripped.startswith("FAIL"):
            print(f"  {stripped}")

    if result.returncode == 0:
        print("  OK  占位提示 / 面板拖动 / 分批高亮 行为全部通过")
    else:
        failures.append("前端行为测试未通过（详见上方 FAIL 行）")
        print("  FAIL 行为测试有失败项")


def main() -> int:
    sys.path.insert(0, str(BASE))
    from app.main import create_app  # noqa: E402 - 需要先注入 sys.path

    # 收集成 {路径: {方法}}：同一个路径可能有多个方法
    # （项目库的 /file 就是 GET + PUT + DELETE），只收集路径会把方法差异漏掉。
    # OpenAPI 里的方法名是小写的，统一转成大写，方便和前端写的 'PUT' 比对。
    operations: dict[str, set[str]] = {}
    for path, item in create_app().openapi()["paths"].items():
        operations[path] = {
            method.upper() for method in item
            if method in {"get", "post", "put", "delete", "patch"}
        }

    for index, page in enumerate(PAGES):
        html = page["html"].read_text(encoding="utf-8")
        js = page["js"].read_text(encoding="utf-8")
        css = page["css"].read_text(encoding="utf-8")

        print()
        print("#" * 72)
        print(f"# {page['name']}：{page['html'].name}")
        print("#" * 72)
        check_syntax(page)
        check_dom_ids(page, html, js)
        check_css_classes(page, html, js, css)
        check_hidden_toggle(page, html, js, css)
        check_api_paths(page, js, operations)
        if index < len(PAGES) - 1:
            print()

    check_js_behavior()

    print()
    print("=" * 72)
    if failures:
        print(f"结论：有 {len(failures)} 项不通过 ×")
        for item in failures:
            print(f"  · {item}")
        return 1
    print("结论：全部检查通过 ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
