"""AI 自动检测验收：真实启动服务，用 HTTP 跑一遍「本地检查 + AI 检测 + 落库」。

两个模式（都会真的起 uvicorn、真的发请求、真的写 SQLite）：
  · 默认：只验证本地静态检查路径（不需要 API Key，任何人都能跑）
       python tools/verify_check.py
  · 加 --ai：额外用 .env 里配置的真实模型跑一轮，验证 AI 深度检测
       python tools/verify_check.py --ai

为什么要单独写这个脚本而不是只靠 pytest：
pytest 用的是进程内 TestClient 与假模型，验证的是逻辑；
本脚本验证的是"学生按教程操作时到底会发生什么"，包含配置、路由、落库一起对不对。
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
PORT = 8314
BASE = f"http://127.0.0.1:{PORT}"

# 三份刻意写坏的示例代码，覆盖三种语言与三类问题
BROKEN_PY = '''def average(scores):
    total = 0
    for i in range(len(scores) + 1):
        total += scores[i]
    return total / len(scores)


def collect(items=[]):
    items.append(1)
    return items


def read(path):
    try:
        handle = open(path, encoding="utf-8")
        return handle.read()
    except:
        return None
'''

BROKEN_C = '''#include <stdio.h>
#include <string.h>

int main(void) {
    char buf[8];
    gets(buf);
    strcpy(buf, "hello world");
    printf("%s\\n", buf)
    return 0;
}
'''

RISKY_JAVA = '''public class ScoreManager {
    public boolean isSame(String a, String b) {
        if (a == "x") {
            return true;
        }
        try {
            Integer.parseInt(a);
        } catch (Exception e) {
        }
        return false;
    }
}
'''


def request(method: str, path: str, body: dict | None = None) -> tuple[int, dict | str]:
    """发一个 HTTP 请求，返回 (状态码, 解析后的内容)。"""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
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


def show(body: dict, indent: str = "       ") -> None:
    """打印一份检测结果，格式尽量贴近学生会看到的样子。"""
    print(f"{indent}评分：{body['score']}（{body['level']}）  AI 参与：{body['ai_available']}")
    print(f"{indent}评分依据：{body['score_reason']}")
    syntax = body["local"]["syntax_error"]
    if syntax:
        print(f"{indent}语法错误：第 {syntax['line']} 行第 {syntax['column']} 列"
              f"（{syntax['tool']}）：{syntax['message']}")
    else:
        print(f"{indent}语法：通过")
    for label, key in (("错误", "errors"), ("风格", "style"), ("风险", "risks")):
        for item in body[key]:
            pos = f"L{item['line']} " if item["line"] else ""
            print(f"{indent}[{label}/{item['severity']}/{item['source']}] "
                  f"{pos}{item['title']}")
            if item["suggestion"]:
                print(f"{indent}    → {item['suggestion'][:90]}")
    for tip in body["advice"]:
        print(f"{indent}学习建议：{tip}")


def main() -> int:
    use_ai = "--ai" in sys.argv
    work = Path(tempfile.mkdtemp())

    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "DATABASE__SQLITE_PATH": str(work / "t.db"),
        "RETRIEVAL__INDEX_DIR": str(work / "index"),
        "REPOSITORY__WORKSPACE_DIR": str(work / "repos"),
        "DOCKER__WORKSPACE_DIR": str(work / "ws"),
        "CLASSIFIER__ROOT": str(work / "codes"),
        "APP__LOG_LEVEL": "WARNING",
        "PYTHONIOENCODING": "utf-8",
    }
    if use_ai:
        # 用真实配置（.env 里的 Key）：直接把项目根目录当工作目录，
        # pydantic-settings 会自己读 .env
        print("=" * 72)
        print("模式：本地检查 + 真实大模型（读 .env 里的 LLM__* 配置）")
        print("=" * 72)
    else:
        env["LLM__API_KEY"] = ""      # 刻意不配模型，验证降级路径
        print("=" * 72)
        print("模式：仅本地静态检查（未配置模型，验证降级路径）")
        print("=" * 72)

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

        # ---------------- 1. Python：语法 + 本地风险规则 ----------------
        print()
        print("=" * 72)
        print("1. Python 代码（含越界循环、可变默认参数、裸 except）")
        print("=" * 72)
        code, body = request(
            "POST", "/api/v1/check/code",
            {"code": BROKEN_PY, "language": "python", "filename": "average.py"},
        )
        assert isinstance(body, dict), body
        show(body)
        titles = " ".join(item["title"] for item in body["risks"] + body["errors"])
        local_ok = "默认参数" in titles and "裸 except" in titles
        checks.append(("Python 本地风险规则命中", code == 200 and local_ok))
        print(f"  {'OK ' if code == 200 and local_ok else 'FAIL'} "
              f"HTTP {code}；可变默认参数与裸 except 均被本地规则指出")

        # ---------------- 2. C：tree-sitter 定位语法错误 ----------------
        print()
        print("=" * 72)
        print("2. C 代码（缺分号 + gets/strcpy 风险）")
        print("=" * 72)
        code, body = request(
            "POST", "/api/v1/check/code",
            {"code": BROKEN_C, "language": "c", "filename": "main.c"},
        )
        assert isinstance(body, dict), body
        show(body)
        syntax = body["local"]["syntax_error"]
        tree_ok = (
            body["syntax_ok"] is False
            and syntax is not None
            and syntax["tool"] == "tree-sitter"
            and syntax["line"] == 8
        )
        checks.append(("C 语法错误由 tree-sitter 精确定位", tree_ok))
        print(f"  {'OK ' if tree_ok else 'FAIL'} 语法错误定位："
              f"{syntax['line'] if syntax else '-'} 行（应为第 8 行，缺分号）")

        risk_titles = " ".join(item["title"] for item in body["risks"])
        risk_ok = "gets" in risk_titles and "strcpy" in risk_titles
        checks.append(("C 危险函数被本地规则指出", risk_ok))
        print(f"  {'OK ' if risk_ok else 'FAIL'} 风险规则：{risk_titles[:80]}")

        score_capped = body["score"] <= 45
        checks.append(("有语法错误时评分被压到 45 以内", score_capped))
        print(f"  {'OK ' if score_capped else 'FAIL'} 评分封顶：{body['score']}（上限 45）")

        # ---------------- 3. Java：字符串比较与空 catch ----------------
        print()
        print("=" * 72)
        print("3. Java 代码（== 比较字符串 + 空 catch）")
        print("=" * 72)
        code, body = request(
            "POST", "/api/v1/check/code",
            {"code": RISKY_JAVA, "language": "Java", "filename": "ScoreManager.java"},
        )
        assert isinstance(body, dict), body
        show(body)
        java_titles = " ".join(item["title"] for item in body["risks"])
        java_ok = "==" in java_titles and "catch" in java_titles
        checks.append(("Java 风险规则命中", java_ok))
        print(f"  {'OK ' if java_ok else 'FAIL'} 风险规则：{java_titles[:80]}")

        # ---------------- 4. 语言推断与错误处理 ----------------
        print()
        print("=" * 72)
        print("4. 语言推断与参数校验")
        print("=" * 72)
        code, inferred = request(
            "POST", "/api/v1/check/code", {"code": "int main(void) { return 0; }\n", "filename": "m.c"}
        )
        ok = code == 200 and isinstance(inferred, dict) and inferred["language"] == "c"
        checks.append(("语言留空时按文件名推断", ok))
        print(f"  {'OK ' if ok else 'FAIL'} filename=m.c -> language={inferred.get('language')}")

        code, bad = request("POST", "/api/v1/check/code", {"code": "package main\n", "language": "go"})
        ok = code == 400
        checks.append(("不支持的语言返回 400", ok))
        print(f"  {'OK ' if ok else 'FAIL'} language=go -> {code} "
              f"{bad.get('detail') if isinstance(bad, dict) else bad}")

        # ---------------- 5. 落库 ----------------
        print()
        print("=" * 72)
        print("5. 检测结果写入 SQLite")
        print("=" * 72)
        code, body = request(
            "POST", "/api/v1/check/code",
            {"code": BROKEN_PY, "language": "python", "filename": "saved.py"},
        )
        assert isinstance(body, dict), body
        record_id = body["record_id"]
        ok = code == 200 and isinstance(record_id, int)
        checks.append(("接口返回落库后的记录 ID", ok))
        print(f"  {'OK ' if ok else 'FAIL'} record_id={record_id}")

        # 直接读 SQLite 核对（sqlite3 是标准库，无需额外依赖）
        import sqlite3

        with sqlite3.connect(work / "t.db") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT filename, language, line_count, score, level, syntax_ok,"
                " error_count, style_count, risk_count, ai_available, model, created_at"
                " FROM code_check_records ORDER BY id"
            ).fetchall()
        print(f"  数据库里共 {len(rows)} 条记录：")
        for row in rows:
            print(f"       #{row['filename']:<18} {row['language']:<7} "
                  f"{row['line_count']:>3} 行  评分={row['score']:<5} "
                  f"错误={row['error_count']} 风格={row['style_count']} "
                  f"风险={row['risk_count']}  AI={row['ai_available']}  "
                  f"{row['created_at'][:19]}")
        saved_ok = len(rows) == 5 and any(row["filename"] == "saved.py" for row in rows)
        checks.append(("四条检测都落库且字段完整", saved_ok))
        print(f"  {'OK ' if saved_ok else 'FAIL'} 落库条数与字段"
              f"（文件名/语言/行数/评分/三类计数/时间）")

        # ---------------- 6. AI 模式下的额外断言 ----------------
        if use_ai:
            print()
            print("=" * 72)
            print("6. 真实模型：深度检测是否给出逻辑层面的结论")
            print("=" * 72)
            code, body = request(
                "POST", "/api/v1/check/code",
                {"code": BROKEN_PY, "language": "python", "filename": "average.py"},
            )
            assert isinstance(body, dict), body
            show(body)
            ai_ok = body["ai_available"] is True and body["ai_score"] is not None
            checks.append(("真实模型参与并给出评分", ai_ok))
            print(f"  {'OK ' if ai_ok else 'FAIL'} ai_available={body['ai_available']} "
                  f"ai_score={body['ai_score']} model={body['model']}")

            ai_issues = [i for i in body["errors"] + body["style"] + body["risks"]
                         if i["source"] == "ai"]
            checks.append(("模型给出了 AI 来源的问题条目", len(ai_issues) >= 1))
            print(f"  {'OK ' if ai_issues else 'FAIL'} AI 来源条目 {len(ai_issues)} 条")
            checks.append(("给出了面向学生的学习建议", len(body["advice"]) >= 1))
            print(f"  {'OK ' if body['advice'] else 'FAIL'} 学习建议 {len(body['advice'])} 条")

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
