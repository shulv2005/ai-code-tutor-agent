"""项目库写入功能验收：在线编辑保存 / 替换 / 删除（回收站）/ 只读模式。

为什么要单独一个验收脚本：
    前四个验收脚本（verify_frontend / verify_files / verify_dragdrop / verify_check …）
    覆盖的都是"读"和"AI 分析"；而这一批新功能会**真的改动学生磁盘上的文件**，
    是整套系统里风险最高的部分。所以这里每一条都既要验证"功能对不对"，
    也要验证"边界守不守得住"：

      场景 1  在线编辑保存：PUT 覆盖内容、换行习惯保留、sha256 乐观锁、越界/超限拦截
      场景 2  文件替换：用另一份内容覆盖同一个路径（前端「替换」按钮走的就是它）
      场景 3  删除：默认移进 .trash 回收站（可找回）、permanent=true 才真删
      场景 4  只读模式：LIBRARY__ALLOW_WRITE=false 时写操作全部 403

运行方式（会自动起临时服务，不需要你手动开）：
    python tools/verify_edit_delete.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PORT = 8319            # 可写模式的服务
PORT_RO = 8320         # 只读模式的服务
BASE = f"http://127.0.0.1:{PORT}"
BASE_RO = f"http://127.0.0.1:{PORT_RO}"

# 单文件上限刻意调小（2000 字节），这样"超过上限返回 413"这一条
# 不用真的造一个几百 KB 的字符串来测；其它用例的文件都远小于它。
MAX_FILE_BYTES = 2000

ORIGINAL_PY = "def add(a, b):\r\n    return a - b\r\n"     # 刻意用 CRLF：验证换行习惯被保留
EDITED_PY = "def add(a, b):\n    return a + b\n"
LF_ONLY_PY = "# 这个文件从头到尾都是 LF\nx = 1\n"

checks: list[tuple[str, bool]] = []


def report(label: str, ok: bool, detail: str = "") -> None:
    """记一条验收结果并打印。"""
    checks.append((label, ok))
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  —— {detail}" if detail else ""))


def request(method: str, path: str, body: dict | None = None,
            base: str = BASE) -> tuple[int, dict | str]:
    """发一个 HTTP 请求，返回 (状态码, 解析后的内容)。"""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{base}{path}", data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
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


def start_server(port: int, env: dict[str, str]) -> subprocess.Popen:
    """起一个临时 uvicorn；数据目录全部指向临时文件夹，不碰开发者的 data/。"""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--port", str(port), "--log-level", "warning"],
        cwd=str(BASE_DIR), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        time.sleep(0.5)
        try:
            request("GET", "/api/v1/health", base=f"http://127.0.0.1:{port}")
            return proc
        except Exception:
            if proc.poll() is not None:
                break
    proc.kill()
    raise RuntimeError(f"端口 {port} 上的服务启动失败")


def build_env(work: Path, library_root: Path, *, allow_write: bool) -> dict[str, str]:
    """构造干净的环境变量：只带系统必需项 + 我们的隔离配置。"""
    return {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PYTHONIOENCODING": "utf-8",
        "DATABASE__SQLITE_PATH": str(work / "app.db"),
        "RETRIEVAL__INDEX_DIR": str(work / "index"),
        "REPOSITORY__WORKSPACE_DIR": str(work / "repos"),
        "DOCKER__WORKSPACE_DIR": str(work / "ws"),
        "LIBRARY__ROOTS": str(library_root),
        "LIBRARY__MAX_FILE_BYTES": str(MAX_FILE_BYTES),
        "LIBRARY__ALLOW_WRITE": "true" if allow_write else "false",
        "LLM__API_KEY": "",              # 项目库这些接口不需要大模型
        "APP__LOG_LEVEL": "WARNING",
    }


def trash_entries(library_root: Path) -> list[Path]:
    """回收站目录里的文件清单（目录不存在时返回空）。"""
    trash = library_root / ".trash"
    return sorted(trash.iterdir()) if trash.is_dir() else []


def as_crlf(text: str) -> bytes:
    """把文本按 CRLF 编码成字节。

    用途：`homework/main.py` 原始文件是 CRLF 的，而保存时会**保留原换行习惯**，
    所以磁盘上写出来的就是 CRLF 版本。断言要按这个口径比，
    否则会误判成"内容没写对"——这条踩过一次。
    """
    return text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8")


def main() -> int:
    work = Path(tempfile.mkdtemp())
    library_root = work / "library"
    (library_root / "homework").mkdir(parents=True)
    (library_root / "homework" / "main.py").write_bytes(ORIGINAL_PY.encode("utf-8"))
    (library_root / "homework" / "crlf.py").write_bytes(ORIGINAL_PY.encode("utf-8"))
    (library_root / "homework" / "lf.py").write_bytes(LF_ONLY_PY.encode("utf-8"))
    (library_root / "homework" / "victim.py").write_bytes("# 会被删除的文件\n".encode())
    (library_root / "homework" / "replace_me.py").write_bytes("# 旧内容\n".encode())

    env = build_env(work, library_root, allow_write=True)
    print("=" * 72)
    print("项目库写入功能验收：在线编辑 / 替换 / 删除 / 只读模式")
    print("=" * 72)

    proc = start_server(PORT, env)
    proc_ro = None
    try:
        # ---------------------------------------------------------------
        # 场景 1：在线编辑保存
        # ---------------------------------------------------------------
        print()
        print("=" * 72)
        print("场景 1  在线编辑保存（PUT /api/v1/library/file）")
        print("=" * 72)

        code, before = request("GET", "/api/v1/library/file?path=homework%2Fmain.py")
        ok = code == 200 and isinstance(before, dict) and before.get("sha256")
        report("读取文件时返回 sha256（前端用它做乐观锁基准）", ok,
               f"sha256={before.get('sha256', '')[:12] if isinstance(before, dict) else '-'}")
        report("读取文件时返回 allow_write（前端据此决定按钮是否可用）",
               isinstance(before, dict) and before.get("allow_write") is True)
        old_sha = before["sha256"] if isinstance(before, dict) else ""

        code, saved = request("PUT", "/api/v1/library/file",
                              {"path": "homework/main.py", "code": EDITED_PY})
        ok = code == 200 and isinstance(saved, dict) and saved.get("line_count") == 2
        report("保存成功并返回新的行数/大小/指纹", ok,
               f"{code} 行数={saved.get('line_count') if isinstance(saved, dict) else '-'} "
               f"新指纹={saved.get('sha256', '')[:12] if isinstance(saved, dict) else '-'}")
        report("保存后指纹与保存前不同（说明内容真的变了）",
               isinstance(saved, dict) and saved.get("sha256") != old_sha)

        on_disk = (library_root / "homework" / "main.py").read_bytes()
        report("磁盘上的文件内容确实被改了（且保留了 CRLF 换行习惯）",
               on_disk == as_crlf(EDITED_PY), repr(on_disk[:40]))
        report("没有留下临时文件（写入是先写 .dsh-tmp 再原子替换）",
               not list((library_root / "homework").glob("*.dsh-tmp")))

        # 换行习惯：CRLF 的文件写回后仍然是 CRLF，LF 的仍然是 LF。
        # 不保留的话，学生用记事本打开会发现"整个文件都变了"。
        request("PUT", "/api/v1/library/file",
                {"path": "homework/crlf.py", "code": "print(1)\nprint(2)\n"})
        crlf_raw = (library_root / "homework" / "crlf.py").read_bytes()
        report("CRLF 文件保存后仍是 CRLF（保留原换行习惯）",
               crlf_raw == b"print(1)\r\nprint(2)\r\n", repr(crlf_raw))

        request("PUT", "/api/v1/library/file",
                {"path": "homework/lf.py", "code": "print(1)\nprint(2)\n"})
        lf_raw = (library_root / "homework" / "lf.py").read_bytes()
        report("LF 文件保存后仍是 LF（不会被悄悄改成 CRLF）",
               lf_raw == b"print(1)\nprint(2)\n", repr(lf_raw))

        # 乐观锁：拿旧指纹去保存必须被拒
        code, conflict = request("PUT", "/api/v1/library/file",
                                 {"path": "homework/main.py", "code": "# 想盖掉\n",
                                  "expected_sha256": old_sha})
        ok = code == 409
        report("用过期指纹保存 → 409（不覆盖别处的改动）", ok,
               conflict.get("detail") if isinstance(conflict, dict) else str(conflict))
        report("409 之后文件内容没有被改动",
               (library_root / "homework" / "main.py").read_bytes() == as_crlf(EDITED_PY))

        # 用当前指纹保存应当成功
        code, again = request("PUT", "/api/v1/library/file",
                              {"path": "homework/main.py", "code": EDITED_PY + "# 又改了一次\n",
                               "expected_sha256": saved.get("sha256") if isinstance(saved, dict) else ""})
        report("用最新指纹保存 → 200（乐观锁不会误伤正常保存）", code == 200,
               again.get("message") if isinstance(again, dict) else str(again))

        # 边界：越界、不存在、二进制、超限
        code, out = request("PUT", "/api/v1/library/file",
                            {"path": "../../escape.py", "code": "x = 1\n"})
        report("越界路径 → 400", code == 400,
               out.get("detail") if isinstance(out, dict) else str(out))

        code, out = request("PUT", "/api/v1/library/file",
                            {"path": "homework/not_here.py", "code": "x = 1\n"})
        report("保存不存在的文件 → 400（新增文件要走入库流程）", code == 400,
               out.get("detail") if isinstance(out, dict) else str(out))

        code, out = request("PUT", "/api/v1/library/file",
                            {"path": "homework/main.py", "code": "x = 1\x00y = 2\n"})
        report("内容含二进制字符（NUL）→ 400，不写进代码文件", code == 400,
               out.get("detail") if isinstance(out, dict) else str(out))

        big = "# 填充\n" * 800            # 远超 2000 字节上限
        code, out = request("PUT", "/api/v1/library/file",
                            {"path": "homework/main.py", "code": big})
        report(f"内容超过 {MAX_FILE_BYTES} 字节上限 → 413", code == 413,
               out.get("detail") if isinstance(out, dict) else str(out))

        # ---------------------------------------------------------------
        # 场景 2：文件替换
        # ---------------------------------------------------------------
        print()
        print("=" * 72)
        print("场景 2  文件替换（用另一份内容覆盖同一个路径，前端「替换」走的接口）")
        print("=" * 72)

        new_content = "def mul(a, b):\n    return a * b\n"
        code, replaced = request("PUT", "/api/v1/library/file",
                                 {"path": "homework/replace_me.py", "code": new_content})
        ok = code == 200 and isinstance(replaced, dict)
        report("替换成功（同一个路径，内容整份换掉）", ok,
               f"{code} {replaced.get('rel_path') if isinstance(replaced, dict) else '-'}")
        report("替换后路径没变、内容变了",
               (library_root / "homework" / "replace_me.py").read_bytes()
               == new_content.encode("utf-8"))
        code, reread = request("GET", "/api/v1/library/file?path=homework%2Freplace_me.py")
        report("替换后读回来就是新内容",
               isinstance(reread, dict) and reread.get("code") == new_content)

        # ---------------------------------------------------------------
        # 场景 3：删除（回收站）
        # ---------------------------------------------------------------
        print()
        print("=" * 72)
        print("场景 3  删除（默认移入 .trash，可找回）")
        print("=" * 72)

        victim = library_root / "homework" / "victim.py"
        before_delete = victim.read_bytes()
        code, deleted = request("DELETE", "/api/v1/library/file?path=homework%2Fvictim.py")
        ok = code == 200 and isinstance(deleted, dict) and deleted.get("permanent") is False
        report("删除成功且报告为「移入回收站」", ok,
               deleted.get("message") if isinstance(deleted, dict) else str(deleted))
        report("原位置的文件已经不在", not victim.exists())

        entries = trash_entries(library_root)
        report("回收站里出现了这个文件", len(entries) == 1,
               ", ".join(p.name for p in entries))
        report("回收站里的内容与删除前完全一致（随时能捞回来）",
               bool(entries) and entries[0].read_bytes() == before_delete)
        report("回收站路径写进了响应（前端要告诉学生去哪找）",
               isinstance(deleted, dict) and str(deleted.get("trash_path", "")).endswith(entries[0].name)
               if entries else False)

        code, scan = request("GET", "/api/v1/library/scan")
        names = [item["rel_path"] for item in scan.get("files", [])] if isinstance(scan, dict) else []
        report("已删除的文件不再出现在项目库列表里", "homework/victim.py" not in names,
               f"当前 {len(names)} 个文件")
        report("回收站目录本身也不会被扫进列表",
               not any(".trash" in name for name in names))

        code, out = request("DELETE", "/api/v1/library/file?path=homework%2Fvictim.py")
        report("重复删除同一个文件 → 400（提示它已经不在了）", code == 400,
               out.get("detail") if isinstance(out, dict) else str(out))

        code, out = request("DELETE", "/api/v1/library/file?path=../../windows%2Fwin.ini")
        report("删除越界路径 → 400", code == 400,
               out.get("detail") if isinstance(out, dict) else str(out))

        # permanent=true 才是真删
        (library_root / "homework" / "gone.py").write_bytes("# 彻底删除\n".encode())
        trash_before = len(trash_entries(library_root))
        code, purged = request(
            "DELETE", "/api/v1/library/file?path=homework%2Fgone.py&permanent=true"
        )
        ok = code == 200 and isinstance(purged, dict) and purged.get("permanent") is True
        report("permanent=true → 彻底删除", ok,
               purged.get("message") if isinstance(purged, dict) else str(purged))
        report("彻底删除不会往回收站里放东西",
               len(trash_entries(library_root)) == trash_before)
        report("彻底删除后文件真的没了", not (library_root / "homework" / "gone.py").exists())

        # ---------------------------------------------------------------
        # 场景 4：只读模式
        # ---------------------------------------------------------------
        print()
        print("=" * 72)
        print("场景 4  只读模式（LIBRARY__ALLOW_WRITE=false）")
        print("=" * 72)

        proc_ro = start_server(PORT_RO, build_env(work, library_root, allow_write=False))

        code, scan_ro = request("GET", "/api/v1/library/scan", base=BASE_RO)
        report("scan 里 allow_write=false（前端据此隐藏编辑/替换/删除按钮）",
               code == 200 and isinstance(scan_ro, dict) and scan_ro.get("allow_write") is False)
        report("只读模式下依然能正常浏览文件（只是不能改）",
               isinstance(scan_ro, dict) and scan_ro.get("total_files", 0) > 0,
               f"{scan_ro.get('total_files') if isinstance(scan_ro, dict) else '-'} 个文件")

        code, out = request("PUT", "/api/v1/library/file",
                            {"path": "homework/main.py", "code": "# 想偷改\n"}, base=BASE_RO)
        report("只读模式下保存 → 403", code == 403,
               out.get("detail") if isinstance(out, dict) else str(out))

        code, out = request("DELETE", "/api/v1/library/file?path=homework%2Fmain.py", base=BASE_RO)
        report("只读模式下删除 → 403", code == 403,
               out.get("detail") if isinstance(out, dict) else str(out))
        report("被拒之后文件仍然完好",
               (library_root / "homework" / "main.py").exists())

        # ---------------------------------------------------------------
        # 场景 5：前端要素（接口通了，还得确认页面真的接上了）
        # ---------------------------------------------------------------
        print()
        print("=" * 72)
        print("场景 5  前端要素静态检查（按钮 / 编辑器 / 弹窗 / 接口调用）")
        print("=" * 72)

        html = (BASE_DIR / "frontend" / "index.html").read_text(encoding="utf-8")
        css = (BASE_DIR / "frontend" / "css" / "style.css").read_text(encoding="utf-8")
        js = (BASE_DIR / "frontend" / "js" / "app.js").read_text(encoding="utf-8")

        report("HTML 有「编辑 / 保存 / 取消」三个按钮与未保存标记",
               all(f'id="{name}"' in html for name in
                   ("codeEditBtn", "codeSaveBtn", "codeCancelBtn", "codeDirtyBadge")))
        report("HTML 有编辑器与行号槽",
               'id="codeEditor"' in html and 'id="editorGutter"' in html)
        report("HTML 有替换用的文件选择框与确认弹窗",
               all(f'id="{name}"' in html for name in
                   ("replaceInput", "confirmModal", "confirmOkBtn", "confirmCancelBtn")))
        report("CSS 定义了编辑器 / 弹窗 / 危险按钮 / 行内操作按钮的样式",
               all(selector in css for selector in
                   (".editor__area", ".modal__box", ".btn--danger", ".files__actions")))
        report("JS 通过 PUT 保存（在线编辑与替换共用）",
               "'/file'" in js and "method: 'PUT'" in js)
        report("JS 通过 DELETE 删除",
               "method: 'DELETE'" in js and "LIBRARY_BASE" in js)
        report("JS 保存时带上乐观锁指纹 expected_sha256", "expected_sha256" in js)
        report("JS 拦下了 Ctrl+S（否则会触发浏览器的「保存网页」）",
               "ctrlKey || event.metaKey" in js and "saveEdit()" in js)
        report("JS 的删除 / 替换按钮走事件委托（data-act）",
               'data-act="replace"' in js and 'data-act="delete"' in js)
        report("JS 在切换文件前会确认未保存的修改",
               "confirmDiscardChanges" in js)

        # ---------------------------------------------------------------
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
        # 两个服务都要收干净：只 terminate 一个的话，另一个会一直占着端口，
        # 下次跑脚本就会"服务启动失败"，还得手动去 kill（踩过）。
        for running in (proc, proc_ro):
            if running is None:
                continue
            running.terminate()
            try:
                running.wait(timeout=10)
            except subprocess.TimeoutExpired:
                running.kill()


if __name__ == "__main__":
    raise SystemExit(main())
