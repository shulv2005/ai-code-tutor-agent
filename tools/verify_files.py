"""文件自动分类验收：真实启动服务，用 HTTP 跑一遍「扫描 → 归档 → 查询」全流程。

与 pytest 用例的区别：
`tests/test_file_classifier.py` 走进程内 TestClient，验证的是逻辑正确；
本脚本**真的起一个 uvicorn 进程**、真的往磁盘写文件、真的调 HTTP 接口，
验证的是"学生按教程操作时到底会发生什么"，包括 config、路由、静态目录一起对不对。

运行：python tools/verify_files.py
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
PORT = 8313
BASE = f"http://127.0.0.1:{PORT}"

C_CODE = "#include <stdio.h>\nint main(void) { return 0; }\n"
JAVA_CODE = "public class A { int f() { return 1; } }\n"
PY_CODE = "def add(a, b):\n    return a + b\n"


def request(method: str, path: str, body: dict | None = None) -> tuple[int, dict | str]:
    """发一个 HTTP 请求，返回 (状态码, 解析后的内容)。"""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
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


def main() -> int:
    work = Path(tempfile.mkdtemp())
    codes = work / "library"
    (codes / "homework").mkdir(parents=True)
    # 故意造出"作业散放 + 子文件夹 + 重名 + 非代码文件"的真实场景
    (codes / "main.c").write_bytes(C_CODE.encode("utf-8"))
    (codes / "utils.h").write_bytes(b"int add(int a, int b);\n")
    (codes / "homework" / "Calculator.java").write_bytes(JAVA_CODE.encode("utf-8"))
    (codes / "homework" / "sort.py").write_bytes(PY_CODE.encode("utf-8"))
    (codes / "readme.txt").write_bytes("说明文档，不是代码\n".encode())
    (codes / "other").mkdir()
    (codes / "other" / "main.c").write_bytes(b"/* the other main */\n")

    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "DATABASE__SQLITE_PATH": str(work / "t.db"),
        "RETRIEVAL__INDEX_DIR": str(work / "index"),
        "REPOSITORY__WORKSPACE_DIR": str(work / "repos"),
        "DOCKER__WORKSPACE_DIR": str(work / "ws"),
        "CLASSIFIER__ROOT": str(codes),
        "LLM__API_KEY": "",
        "APP__LOG_LEVEL": "WARNING",
        "PYTHONIOENCODING": "utf-8",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT),
         "--log-level", "warning"],
        cwd=str(BASE_DIR), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    checks: list[tuple[str, bool]] = []
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

        print("=" * 72)
        print("1. 预演模式（dry_run）：只说不做")
        print("=" * 72)
        code, plan = request("POST", "/api/v1/files/scan", {"dry_run": True})
        ok = (
            code == 200
            and isinstance(plan, dict)
            and plan["dry_run"] is True
            and plan["inserted"] == 0
            and (codes / "main.c").is_file()          # 文件没被搬走
        )
        checks.append(("预演不移动文件、不写库", ok))
        print(f"  {'OK ' if ok else 'FAIL'} {code}  计划归档 {plan.get('moved')} 个，"
              f"写入数据库 {plan.get('inserted')} 条")
        for item in (plan.get("files") or [])[:5]:
            print(f"       {item['source_path']}  ->  {item['target_path']}  "
                  f"({item['language_label']}, {item['action']})")

        print()
        print("=" * 72)
        print("2. 正式扫描：按语言归档")
        print("=" * 72)
        code, result = request("POST", "/api/v1/files/scan", {})
        assert isinstance(result, dict), result
        # 临时目录里放了 3 个 C（main.c、utils.h、other/main.c）、1 个 Java、1 个 Python、
        # 1 个 .txt（未知），共 6 个
        expected = {"C": 3, "Java": 1, "Python": 1, "未知": 1}
        labels = {plan["language_labels"][k]: v for k, v in (result.get("counts") or {}).items()}
        ok = code == 200 and labels == expected
        checks.append(("扫描分类结果符合预期", ok))
        print(f"  {'OK ' if ok else 'FAIL'} {code}  分类={labels}  "
              f"归档={result.get('moved')} 未知={result.get('unknown')}  "
              f"入库={result.get('inserted')} 条  耗时={result.get('duration_ms')}ms")

        # 文件真的到了对应目录，且原位置已经空了
        layout_ok = all(
            [
                (codes / "c" / "main.c").is_file(),
                (codes / "c" / "utils.h").is_file(),
                (codes / "java" / "Calculator.java").is_file(),
                (codes / "python" / "sort.py").is_file(),
                not (codes / "main.c").exists(),
                (codes / "readme.txt").is_file(),       # 未知文件留在原地
            ]
        )
        checks.append(("文件已归档到 c/ java/ python/", layout_ok))
        print(f"  {'OK ' if layout_ok else 'FAIL'} 目录实际内容：")
        for folder in ("c", "java", "python", "unknown"):
            target = codes / folder
            names = sorted(p.name for p in target.iterdir()) if target.is_dir() else []
            print(f"       {folder + '/':<9} {names or '（不存在）'}")

        # 重名文件不能互相覆盖
        dup_ok = (codes / "c" / "main.c").is_file() and (codes / "c" / "main_1.c").is_file()
        checks.append(("同名文件自动加后缀、不覆盖", dup_ok))
        print(f"  {'OK ' if dup_ok else 'FAIL'} c/ 下两个 main："
              f"{sorted(p.name for p in (codes / 'c').iterdir())}")

        print()
        print("=" * 72)
        print("3. 幂等：再扫一次不应重复搬动")
        print("=" * 72)
        code, again = request("POST", "/api/v1/files/scan", {})
        assert isinstance(again, dict), again
        ok = code == 200 and again["moved"] == 0 and again["total"] == 1
        checks.append(("重复扫描无副作用", ok))
        print(f"  {'OK ' if ok else 'FAIL'} {code}  第二次：文件={again['total']} "
              f"归档={again['moved']}（应为 0，归档目录被跳过）")
        nested = (codes / "c" / "c").exists()
        checks.append(("未出现 c/c/ 套娃目录", not nested))
        print(f"  {'OK ' if not nested else 'FAIL'} c/c/ 套娃目录：{'不存在' if not nested else '出现了！'}")

        print()
        print("=" * 72)
        print("4. GET /api/v1/files/list：按语言查询 SQLite 台账")
        print("=" * 72)
        code, listing = request("GET", "/api/v1/files/list")
        assert isinstance(listing, dict), listing
        ok = code == 200 and listing["total"] == 6 and listing["by_language"]["c"] == 3
        checks.append(("列表返回全部记录", ok))
        print(f"  {'OK ' if ok else 'FAIL'} {code}  共 {listing['total']} 条  "
              f"分类={listing['by_language']}")
        for row in listing["items"]:
            print(f"       #{row['id']} {row['filename']:<18} {row['language_label']:<6} "
                  f"{row['path']:<22} {row['size_bytes']:>4}B  "
                  f"入库={row['created_at'][:19]}  在磁盘上={row['exists']}")

        code, only_java = request("GET", "/api/v1/files/list?language=java")
        assert isinstance(only_java, dict), only_java
        ok = code == 200 and only_java["total"] == 1
        checks.append(("按语言筛选", ok))
        print(f"  {'OK ' if ok else 'FAIL'} {code}  只看 java -> "
              f"{[r['filename'] for r in only_java['items']]}")

        code, bad = request("GET", "/api/v1/files/list?language=cobol")
        ok = code == 400
        checks.append(("非法语言返回 400", ok))
        print(f"  {'OK ' if ok else 'FAIL'} {code}  语言=cobol -> "
              f"{bad.get('detail') if isinstance(bad, dict) else bad}")

        print()
        print("=" * 72)
        print("5. 安全边界：不允许扫描根目录之外")
        print("=" * 72)
        code, blocked = request("POST", "/api/v1/files/scan", {"directory": "../.."})
        ok = code == 400
        checks.append(("越界目录被拒绝", ok))
        print(f"  {'OK ' if ok else 'FAIL'} {code}  directory=../.. -> "
              f"{blocked.get('detail') if isinstance(blocked, dict) else blocked}")

        code, sub = request("POST", "/api/v1/files/scan", {"directory": "homework"})
        assert isinstance(sub, dict), sub
        # 子目录里的文件已经在上一步归档走了，这里应该扫到 0 个
        ok = code == 200
        checks.append(("指定子目录可正常扫描", ok))
        print(f"  {'OK ' if ok else 'FAIL'} {code}  directory=homework -> "
              f"文件={sub['total']}（已归档的不重复处理）")

        print()
        print("=" * 72)
        passed = sum(1 for _, ok in checks if ok)
        print(f"验收结果：{passed}/{len(checks)} 项通过")
        print("=" * 72)
        for label, ok in checks:
            print(f"  {'OK  ' if ok else 'FAIL'} {label}")
        return 0 if passed == len(checks) else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
