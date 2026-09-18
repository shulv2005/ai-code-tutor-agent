"""需求一 + 需求二 验收：拖拽入库、占位提示、去重（真实 HTTP 全流程）。

这个脚本把用户在课堂上会做的操作完整跑一遍，用真实的 HTTP 请求验证：

  场景 1  拖 .py 到「本地项目库」→ 入库 + 归档 + 项目库列表能看到 + 分类信息正确
  场景 2  拖 .c 到「代码预览区」 → 只预览，**不入库**（项目库文件数不变）
  场景 3  重复拖同一个文件        → 「文件已存在」，不重复添加
  场景 4  同名不同内容            → 另存为新名字，不覆盖
  场景 5  不支持的类型            → 明确拒绝并说明支持哪些

另外做两项静态检查（前端行为的"确定部分"）：
  · HTML 里有 #code-placeholder 且用 CSS 类控制显隐
  · JS 里有 hasCode 状态与三个拖拽绑定

运行：python tools/verify_dragdrop.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PORT = 8318
BASE = f"http://127.0.0.1:{PORT}"

PY_CODE = "def add(a, b):\n    return a + b\n\nprint(add(1, 2))\n"
C_CODE = "#include <stdio.h>\n\nint main(void) {\n    return 0;\n}\n"

checks: list[tuple[str, bool]] = []


def report(label: str, ok: bool, detail: str = "") -> None:
    """记一条验收结果并打印。"""
    checks.append((label, ok))
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  —— {detail}" if detail else ""))


def request(method: str, path: str, body: bytes | None = None, content_type: str | None = None):
    """发一个 HTTP 请求，返回 (状态码, 解析后的内容)。

    body 直接给字节，这样才能手工拼 multipart/form-data（不引入额外依赖）。
    """
    req = urllib.request.Request(f"{BASE}{path}", data=body, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(payload)
            except json.JSONDecodeError:
                return resp.status, payload
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(payload)
        except json.JSONDecodeError:
            return exc.code, payload


def upload_form(filename: str, content: bytes) -> tuple[bytes, str]:
    """手工拼一个 multipart/form-data 请求体（模拟浏览器拖拽上传的文件）。"""
    boundary = f"----DSHDrag{uuid.uuid4().hex[:8]}"
    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: application/octet-stream\r\n\r\n",
        content,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def library_files() -> list[dict]:
    """取当前项目库的文件列表（前端列表用的就是这个接口）。"""
    code, data = request("GET", "/api/v1/library/scan")
    return data.get("files", []) if isinstance(data, dict) else []


def main() -> int:
    work = Path(tempfile.mkdtemp())
    library_root = work / "library"
    library_root.mkdir()

    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "DATABASE__SQLITE_PATH": str(work / "t.db"),
        "RETRIEVAL__INDEX_DIR": str(work / "index"),
        "REPOSITORY__WORKSPACE_DIR": str(work / "repos"),
        "DOCKER__WORKSPACE_DIR": str(work / "ws"),
        # 分类器与项目库指向同一个目录（生产默认也是同一个），
        # 这样"拖进去 → 归档 → 列表能看到"整条链路才是通的
        "CLASSIFIER__ROOT": str(library_root),
        "LIBRARY__ROOTS": str(library_root),
        "LLM__API_KEY": "",
        "APP__LOG_LEVEL": "WARNING",
        "PYTHONIOENCODING": "utf-8",
    }
    print("=" * 72)
    print("需求一 + 需求二验收：拖拽入库 / 占位提示 / 去重")
    print("=" * 72)

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT),
         "--log-level", "warning"],
        cwd=str(BASE_DIR), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        for _ in range(60):
            time.sleep(0.5)
            try:
                request("GET", "/api/v1/health")
                break
            except Exception:
                if proc.poll() is not None:
                    print("服务启动失败")
                    return 1

        # ---------------- 场景 1：拖 .py 进项目库 ----------------
        print()
        print("=" * 72)
        print("场景 1  拖一个 .py 到「② 本地项目库」→ 入库 + 归档 + 列表可见")
        print("=" * 72)
        before = library_files()
        body, ctype = upload_form("homework1.py", PY_CODE.encode("utf-8"))
        status, uploaded = request("POST", "/api/v1/files/upload", body, ctype)
        assert isinstance(uploaded, dict), uploaded
        print(f"    POST /files/upload -> {status}  {uploaded.get('rel_path')}  "
              f"({uploaded.get('language_label')})")
        report("上传成功（201）", status == 201, uploaded.get("note", ""))
        report("返回的语言识别正确", uploaded.get("language") == "python")

        status, scanned = request("POST", "/api/v1/files/scan", b"{}", "application/json")
        assert isinstance(scanned, dict), scanned
        print(f"    POST /files/scan   -> {status}  归档 {scanned.get('moved')} 个，"
              f"入库 {scanned.get('inserted')} 条")
        report("归档接口正常", status == 200 and scanned.get("moved", 0) >= 1)

        after = library_files()
        paths = [item["rel_path"] for item in after]
        print(f"    项目库列表：{paths}")
        report("项目库列表新增了该文件", "python/homework1.py" in paths, "归档后在 python/ 下")
        report("列表长度增加", len(after) == len(before) + 1)

        entry = next((item for item in after if item["rel_path"] == "python/homework1.py"), None)
        report(
            "分类信息正确（语言 / 行数）",
            bool(entry) and entry["language"] == "python" and entry["line_count"] == 4,
            f"{entry['language_label']} {entry['line_count']} 行" if entry else "未找到",
        )

        # 前端「③ 自动分类结果」用的是 /analyze，这里顺带验证它能识别语言与行数
        form = (
            b"------DSH\r\n"
            b'Content-Disposition: form-data; name="filename"\r\n\r\nhomework1.py\r\n'
            b"------DSH\r\n"
            b'Content-Disposition: form-data; name="code"\r\n\r\n' + PY_CODE.encode() + b"\r\n"
            b"------DSH--\r\n"
        )
        status, analyzed = request(
            "POST", "/api/v1/tutor/analyze", form, "multipart/form-data; boundary=----DSH"
        )
        report(
            "自动分类结果可用（语言/行数/字符数）",
            status == 200
            and analyzed.get("language") == "python"
            and analyzed.get("line_count") == 4
            and analyzed.get("char_count") == len(PY_CODE),
            f"{analyzed.get('language_label')} {analyzed.get('line_count')} 行 "
            f"{analyzed.get('char_count')} 字符",
        )

        # ---------------- 场景 2：拖 .c 进预览区（不入库）----------------
        print()
        print("=" * 72)
        print("场景 2  拖一个 .c 到「③ 代码预览区」→ 只预览，不入库")
        print("=" * 72)
        count_before = len(library_files())
        # 预览路径完全不经过 /files/*，前端只调 /analyze；这里验证"不入库"这一点
        form = (
            b"------DSH\r\n"
            b'Content-Disposition: form-data; name="filename"\r\n\r\npreview_only.c\r\n'
            b"------DSH\r\n"
            b'Content-Disposition: form-data; name="code"\r\n\r\n' + C_CODE.encode() + b"\r\n"
            b"------DSH--\r\n"
        )
        status, analyzed = request(
            "POST", "/api/v1/tutor/analyze", form, "multipart/form-data; boundary=----DSH"
        )
        report(
            "预览路径识别为 C",
            status == 200 and analyzed.get("language") == "c",
            f"{analyzed.get('language_label')} 行数={analyzed.get('line_count')}",
        )
        count_after = len(library_files())
        report("预览没有往项目库加文件", count_after == count_before,
               f"{count_before} -> {count_after}")
        report("服务器目录里也没有这个文件", not (library_root / "preview_only.c").exists())

        # ---------------- 场景 3：重复拖同一个文件 ----------------
        print()
        print("=" * 72)
        print("场景 3  再次拖入同一个 .py → 「文件已存在」，不重复添加")
        print("=" * 72)
        before_dup = len(library_files())
        body, ctype = upload_form("homework1.py", PY_CODE.encode("utf-8"))
        status, dup = request("POST", "/api/v1/files/upload", body, ctype)
        assert isinstance(dup, dict), dup
        print(f"    POST /files/upload -> {status}  {dup.get('detail')}")
        report("返回 409", status == 409)
        report("提示里有「文件已存在」", "文件已存在" in str(dup.get("detail", "")))
        report("项目库没有多出文件", len(library_files()) == before_dup)

        # ---------------- 场景 4：同名不同内容 ----------------
        print()
        print("=" * 72)
        print("场景 4  同名但内容不同 → 另存为新名字，不覆盖")
        print("=" * 72)
        other = PY_CODE.replace("add", "plus")
        body, ctype = upload_form("homework1.py", other.encode("utf-8"))
        status, renamed = request("POST", "/api/v1/files/upload", body, ctype)
        assert isinstance(renamed, dict), renamed
        print(f"    POST /files/upload -> {status}  保存为 {renamed.get('filename')}")
        report("返回 201 且已改名", status == 201 and renamed.get("renamed") is True)
        report("文件名带 _1 后缀（与归档目录里的同名文件也不冲突）",
               renamed.get("filename") == "homework1_1.py")
        # 第一份已经被归档到 python/ 下，内容必须原样保留
        report(
            "归档的那份内容没被覆盖",
            (library_root / "python" / "homework1.py").read_text(encoding="utf-8") == PY_CODE,
        )

        # ---------------- 场景 5：不支持的类型 ----------------
        print()
        print("=" * 72)
        print("场景 5  拖入 .txt → 明确拒绝并说明支持范围")
        print("=" * 72)
        body, ctype = upload_form("notes.txt", b"hello")
        status, rejected = request("POST", "/api/v1/files/upload", body, ctype)
        assert isinstance(rejected, dict), rejected
        print(f"    POST /files/upload -> {status}  {rejected.get('detail')}")
        report("返回 400 且说明只支持三种语言",
               status == 400 and "只支持" in str(rejected.get("detail", "")))

        # ---------------- 静态检查：需求一的 DOM / CSS / JS ----------------
        print()
        print("=" * 72)
        print("场景 6  占位提示与拖拽绑定的静态检查（需求一第 1/2/5 条）")
        print("=" * 72)
        html = (BASE_DIR / "frontend" / "index.html").read_text(encoding="utf-8")
        css = (BASE_DIR / "frontend" / "css" / "style.css").read_text(encoding="utf-8")
        js = (BASE_DIR / "frontend" / "js" / "app.js").read_text(encoding="utf-8")

        report(
            "HTML 里有 #code-placeholder 且 class=placeholder",
            'id="code-placeholder"' in html and 'class="placeholder"' in html,
        )
        report(
            "CSS 有 .placeholder 与 .placeholder.hidden",
            ".placeholder {" in css and ".placeholder.hidden" in css,
        )
        report("JS 里有 hasCode 状态变量", "hasCode" in js)
        report("JS 用 classList 控制显隐（不直接改 display）",
               "classList.toggle('hidden'" in js)
        report("三个 AI 按钮在没有代码时禁用（可点条件含 hasCode）",
               "state.aiAvailable && state.hasCode" in js)
        report("三个拖拽区都绑定了处理器",
               "bindDropzone()" in js and "bindLibraryDropzone()" in js
               and "bindCodeDropzone()" in js)
        report("清空按钮存在（占位提示会重新出现）",
               'id="clearCodeBtn"' in html and "clearCode" in js)

        # ---------------- 静态检查：需求三 / 需求四 的要素 ----------------
        # 行为本身由 tools/check_frontend_logic.mjs 真跑 app.js 验证（100 项），
        # 这里只确认关键要素都在，两道检查互相补位。
        print()
        print("=" * 72)
        print("场景 7  分隔条与懒加载（需求三 / 需求四）")
        print("=" * 72)
        report("HTML 里有三根分隔条",
               all(f'id="{name}"' in html for name in
                   ("resizerLeft", "resizerRight", "resizerCode")))
        # 数 role 之前要先去掉 HTML 注释：注释里也会提到 role="separator"，
        # 直接 count 会把说明文字也算成一根分隔条（这个坑刚踩过）
        html_without_comments = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
        # 先把计数取出来再用：Python 3.11 的 f-string 表达式里不允许出现反斜杠
        separator_count = html_without_comments.count('role="separator"')
        report("分隔条带 role=separator（可聚焦、可键盘调）",
               separator_count == 3, f"实际 {separator_count} 根")
        report("CSS 里有 col-resize / row-resize 鼠标样式",
               "col-resize" in css and "row-resize" in css)
        report("CSS 用 CSS 变量做栅格（拖动只改一个变量）",
               "var(--sidebar-w)" in css and "var(--result-w)" in css)
        report("CSS 里中间栏最小宽度 400px", "--main-min: 400px" in css)
        report("JS 里最小宽度常量符合需求（200 / 400 / 280）",
               "sidebar: 200" in js and "main: 400" in js and "result: 280" in js)
        report("JS 有懒加载阈值 500 行与 requestIdleCallback",
               "LAZY_HIGHLIGHT_THRESHOLD = 500" in js and "requestIdleCallback" in js)
        report("JS 会在切换文件时取消未完成的高亮任务",
               "cancelPendingHighlight" in js)
        report("JS 会避开块注释中间切分（防止高亮串色）",
               "safeSplitPoint" in js)

        # ---------------- 静态检查：项目库文件夹路径常显 ----------------
        # 用户反馈"我需要知道本地项目库在哪个文件夹"，所以路径必须一直显示，
        # 而不是只在扫不到文件时才出现（那种做法在有文件时反而看不到关键信息）。
        print()
        print("=" * 72)
        print("场景 8  项目库文件夹路径常显（一眼看到代码存在哪）")
        print("=" * 72)
        report("HTML 有常显的路径行与复制按钮",
               'id="libraryPathValue"' in html and 'id="copyLibraryPath"' in html)
        report("路径行在拖拽区之后、列表之前（位置符合阅读顺序）",
               html.index('id="libraryDropzone"') < html.index('id="libraryPathValue"')
               < html.index('id="libraryList"'))
        report("CSS 定义了 .library__path 且长路径用省略号截断",
               ".library__path {" in css and "text-overflow: ellipsis" in css)
        report("JS 无条件调用 renderLibraryPath（不在「零文件」分支里）",
               "renderLibraryPath(roots);" in js)
        report("JS 用 navigator.clipboard 实现一键复制",
               "clipboard.writeText" in js)
        report("已删掉不再需要的 fetchDefaultLibraryPath（死代码清理）",
               "fetchDefaultLibraryPath" not in js)

        print()
        print("=" * 72)
        passed = sum(1 for _, ok in checks if ok)
        print(f"验收结果：{passed}/{len(checks)} 项通过")
        print("=" * 72)
        for label, ok in checks:
            if not ok:
                print(f"  FAIL  {label}")
        return 0 if passed == len(checks) else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
