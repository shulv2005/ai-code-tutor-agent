"""代码注释生成验收：真实启动服务，用 HTTP 跑一遍「三层注释 + 复检 + 落库」。

两个模式（都会真的起 uvicorn、真的发请求、真的写 SQLite）：
  · 默认：只验证本地路径（不需要 API Key）
       python tools/verify_comment.py
  · 加 --ai：用 .env 里配置的真实模型跑一轮，验证三层注释与复检
       python tools/verify_comment.py --ai

重点验证三件事（注释类功能最容易糊弄过去的地方）：
  1. 三层注释**真的都加上了**（文件级 / 函数级 / 行内），且符合语言规范；
  2. **代码逻辑没有被改动**——这是本地 AST 比对给出的客观结论，
     不是"模型说它没改"；
  3. 原代码与带注释代码都进了 SQLite，学生能回看对比。
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PORT = 8316
BASE = f"http://127.0.0.1:{PORT}"

PLAIN_PY = '''def average(scores):
    total = 0
    for score in scores:
        total += score
    return total / len(scores)


def highest(scores):
    best = scores[0]
    for score in scores:
        if score > best:
            best = score
    return best


print(average([88, 92, 79]), highest([88, 92, 79]))
'''

PLAIN_C = '''#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main(void) {
    printf("%d\\n", add(1, 2));
    return 0;
}
'''

PLAIN_JAVA = '''public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
'''

CJK = re.compile(r"[\u4e00-\u9fff]")


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
    """打印一份注释生成结果，尽量贴近学生看到的样子。"""
    check = body["verification"]
    print(f"{indent}语言：{body['language_label']}  AI 参与：{body['ai_available']}  "
          f"总评：{body['summary']}")
    print(f"{indent}复检：语法{'通过' if check['syntax_ok'] else '不通过'} / "
          f"代码未改动={check['code_unchanged']} / "
          f"函数注释 {check['functions_covered']}/{check['functions_total']}"
          f"（覆盖率 {check['coverage_ratio']:.0%}）/ "
          f"文件级注释={check['file_comment']}")
    print(f"{indent}注释行：{check['comment_lines_before']} -> "
          f"{check['comment_lines_after']}（新增 {check['added_comment_lines']}），"
          f"其中行内注释 {check['inline_comment_lines']} 行")
    print(f"{indent}说明：{check['note']}")
    if body["warnings"]:
        for item in body["warnings"]:
            print(f"{indent}[提示] {item}")


def preview(code: str, lines: int = 12, indent: str = "       ") -> None:
    """打印带注释代码的前若干行，让人直接看到效果。"""
    for row in code.splitlines()[:lines]:
        print(f"{indent}| {row}")
    if len(code.splitlines()) > lines:
        print(f"{indent}| ...（共 {len(code.splitlines())} 行）")


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
    print("=" * 72)
    if use_ai:
        print("模式：本地分析 + 真实大模型写注释（读 .env 里的 LLM__* 配置）")
    else:
        env["LLM__API_KEY"] = ""      # 刻意不配模型，验证降级路径
        print("模式：仅本地路径（未配置模型，验证降级行为）")
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

        # ---------------- 1. Python：三层注释 ----------------
        print()
        print("=" * 72)
        print("1. Python 代码（两个函数 + 顶层调用）")
        print("=" * 72)
        code, body = request(
            "POST", "/api/v1/comment/generate",
            {"code": PLAIN_PY, "language": "python", "filename": "score.py"},
        )
        assert isinstance(body, dict), body
        show(body)
        preview(body["commented_code"])

        if use_ai:
            check = body["verification"]
            # ① 文件级注释
            checks.append(("文件级注释已添加", check["file_comment"] is True))
            # ② 函数级注释：两个函数都要有
            checks.append(
                (f"函数注释覆盖 2/2（实际 {check['functions_covered']}/{check['functions_total']}）",
                 check["functions_covered"] == 2 and check["functions_total"] == 2)
            )
            # ③ 行内注释：至少要有一条针对关键逻辑的
            checks.append(("有行内注释（解释关键逻辑）", check["inline_comment_lines"] >= 1))
            # ④ Python 规范：文档字符串
            docstring_ok = body["commented_code"].count('"""') >= 6 or "'''" in body["commented_code"]
            checks.append(("使用 Python 文档字符串（docstring）", docstring_ok))
            # ⑤ 中文注释
            cjk_ok = len(CJK.findall(body["commented_code"])) >= 40
            checks.append(("注释是中文且内容充足", cjk_ok))
            # ⑥ 代码逻辑没被改动（本地 AST 比对）
            unchanged = check["code_unchanged"]
            checks.append(("代码逻辑未被改动（AST 比对）", unchanged is True))
            # ⑦ 独立复核：把模型给的代码交给 ast 跑一遍
            try:
                ast.parse(body["commented_code"])
                parse_ok = True
            except SyntaxError:
                parse_ok = False
            checks.append(("生成的代码仍可被 ast 解析", parse_ok))
        else:
            # 无模型时：必须原样返回原代码 + 说明原因
            keep_ok = body["commented_code"] == PLAIN_PY and bool(body["note"])
            checks.append(("无模型时原样返回原代码并说明原因", keep_ok))
            print(f"  {'OK ' if keep_ok else 'FAIL'} 原代码原样返回 + 提示："
                  f"{body['note'][:60]}")

        # ---------------- 2. C：块注释规范 ----------------
        print()
        print("=" * 72)
        print("2. C 代码（检查函数上方块注释与文件级注释）")
        print("=" * 72)
        code, body = request(
            "POST", "/api/v1/comment/generate",
            {"code": PLAIN_C, "language": "c", "filename": "add.c"},
        )
        assert isinstance(body, dict), body
        show(body)
        preview(body["commented_code"])

        if use_ai:
            commented_c = body["commented_code"]
            checks.append(("C 使用 /* */ 块注释", "/*" in commented_c))
            checks.append(
                ("C 函数注释覆盖 2/2",
                 body["verification"]["functions_covered"] == 2)
            )
            checks.append(
                ("C 代码逻辑未被改动",
                 body["verification"]["code_unchanged"] is True)
            )

        # ---------------- 3. Java：Javadoc ----------------
        print()
        print("=" * 72)
        print("3. Java 代码（检查 Javadoc 的 @param / @return）")
        print("=" * 72)
        code, body = request(
            "POST", "/api/v1/comment/generate",
            {"code": PLAIN_JAVA, "language": "Java", "filename": "Calculator.java"},
        )
        assert isinstance(body, dict), body
        show(body)
        preview(body["commented_code"])

        if use_ai:
            commented_java = body["commented_code"]
            checks.append(("Java 使用 Javadoc（/** */）", "/**" in commented_java))
            checks.append(("@param 说明参数", "@param" in commented_java))
            checks.append(("@return 说明返回值", "@return" in commented_java))
            checks.append(
                ("Java 代码逻辑未被改动",
                 body["verification"]["code_unchanged"] is True)
            )

        # ---------------- 4. 参数校验 ----------------
        print()
        print("=" * 72)
        print("4. 语言推断与参数校验")
        print("=" * 72)
        code, inferred = request(
            "POST", "/api/v1/comment/generate",
            {"code": "int main(void) { return 0; }\n", "filename": "m.c"},
        )
        ok = code == 200 and isinstance(inferred, dict) and inferred["language"] == "c"
        checks.append(("语言留空时按文件名推断", ok))
        print(f"  {'OK ' if ok else 'FAIL'} filename=m.c -> language="
              f"{inferred.get('language') if isinstance(inferred, dict) else inferred}")

        code, bad = request(
            "POST", "/api/v1/comment/generate", {"code": "package main\n", "language": "go"}
        )
        ok = code == 400
        checks.append(("不支持的语言返回 400", ok))
        print(f"  {'OK ' if ok else 'FAIL'} language=go -> {code} "
              f"{bad.get('detail') if isinstance(bad, dict) else bad}")

        # ---------------- 5. 历史：对比学习 ----------------
        print()
        print("=" * 72)
        print("5. 生成历史写入 SQLite 且能回看对比")
        print("=" * 72)
        code, listing = request("GET", "/api/v1/comment/history")
        assert isinstance(listing, dict), listing
        print(f"  历史列表：共 {listing['total']} 条")
        for item in listing["items"]:
            print(f"       #{item['id']} {item['filename']:<16} {item['language']:<7} "
                  f"新增注释={item['added_comment_lines']:>2} 行  "
                  f"函数注释={item['functions_covered']}/{item['functions_total']}  "
                  f"代码未改={item['code_unchanged']}  "
                  f"{item['created_at'][:19]}")
        list_ok = code == 200 and listing["total"] >= 3
        checks.append(("历史列表能读到历次记录", list_ok))
        print(f"  {'OK ' if list_ok else 'FAIL'} GET /comment/history -> {code}")

        first_id = listing["items"][-1]["id"] if listing["items"] else None
        code, detail = request("GET", f"/api/v1/comment/history/{first_id}")
        assert isinstance(detail, dict), detail
        detail_ok = (
            code == 200
            and bool(detail["original_code"])
            and bool(detail["commented_code"])
        )
        checks.append(("历史详情含原代码与带注释代码", detail_ok))
        print(f"  {'OK ' if detail_ok else 'FAIL'} 详情 #{first_id}："
              f"原代码 {len(detail['original_code'].splitlines())} 行 / "
              f"带注释 {len(detail['commented_code'].splitlines())} 行")

        # 直接查库核对
        import sqlite3

        with sqlite3.connect(work / "t.db") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT filename, language, added_comment_lines, functions_total,"
                " functions_covered, code_unchanged, verified, ai_available,"
                " length(original_code) AS orig_len, length(commented_code) AS new_len"
                " FROM comment_records ORDER BY id"
            ).fetchall()
        print(f"  数据库 comment_records 共 {len(rows)} 条：")
        for row in rows:
            print(f"       #{row['filename']:<16} 原代码 {row['orig_len']:>4} 字符 / "
                  f"带注释 {row['new_len']:>4} 字符  新增注释={row['added_comment_lines']} "
                  f"复检={row['verified']} AI={row['ai_available']}")
        db_ok = len(rows) == len(listing["items"]) and all(
            row["orig_len"] and row["new_len"] for row in rows
        )
        checks.append(("库里同时存了原代码与带注释代码", db_ok))
        print(f"  {'OK ' if db_ok else 'FAIL'} 每条记录都同时有两份代码全文")

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
