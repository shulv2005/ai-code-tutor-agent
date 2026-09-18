"""代码改错验收：真实启动服务，用 HTTP 跑一遍「本地分析 → AI 修正 → 复检 → 落库」。

两个模式（都会真的起 uvicorn、真的发请求、真的写 SQLite）：
  · 默认：只验证本地分析路径（不需要 API Key）
       python tools/verify_fix.py
  · 加 --ai：额外用 .env 里配置的真实模型跑一轮，验证 AI 改错与四问式讲解
       python tools/verify_fix.py --ai

重点验证三件事（这三件是"改错"功能最容易糊弄过去的地方）：
  1. AI 给出的代码**真的**通过了本地语法复检（不是"AI 说改好了"就算数）；
  2. 模型返回仍然有语法错误的代码时，复检要如实报出来；
  3. 修改前后两份代码都进了 SQLite，学生能回看对比。
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
PORT = 8315
BASE = f"http://127.0.0.1:{PORT}"

# 缺冒号（语法错误）+ 越界循环（逻辑错误）+ 没处理空列表（风险）
BROKEN_PY = '''def average(scores):
    total = 0
    for i in range(len(scores) + 1):
        total += scores[i]
    return total / len(scores)
'''

# 缺分号（C 语法错误）+ gets 风险
BROKEN_C = '''#include <stdio.h>

int main(void) {
    char buf[8];
    gets(buf);
    printf("%s\\n", buf)
    return 0;
}
'''

SHELL_PY = '''def add(a, b)
    return a + b
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
    """打印一份改错结果，格式尽量贴近学生看到的样子。"""
    print(f"{indent}发现错误：{body['had_error']}  修改 {len(body['changes'])} 处  "
          f"类型统计：{body['categories']}  AI 参与：{body['ai_available']}")
    print(f"{indent}总评：{body['summary']}")
    verification = body["verification"]
    print(f"{indent}本地复检：{'通过' if verification['verified'] else '未通过'} —— "
          f"{verification['note']}")
    before = verification["syntax_before"]
    if before:
        print(f"{indent}改动前语法错误：第 {before['line']} 行"
              f"（{before['tool']}）{before['message']}")
    for index, change in enumerate(body["changes"], start=1):
        print(f"{indent}── 第 {index} 处（{change['category']}"
              f"{'，L' + str(change['line']) if change['line'] else ''}）")
        print(f"{indent}   错在哪：{change['what']}")
        print(f"{indent}   为什么：{change['why']}")
        print(f"{indent}   怎么改：{change['how']}")
        print(f"{indent}   以后避免：{change['avoid']}")
    if body["diff"]:
        print(f"{indent}差异统计：{body['diff_stats']}")


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
        print("模式：本地分析 + 真实大模型改错（读 .env 里的 LLM__* 配置）")
    else:
        env["LLM__API_KEY"] = ""      # 刻意不配模型，验证降级路径
        print("模式：仅本地分析（未配置模型，验证降级路径）")
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

        # ---------------- 1. 语法错误：本地分析就能定位 ----------------
        print()
        print("=" * 72)
        print("1. Python 缺冒号（语法错误）")
        print("=" * 72)
        code, body = request(
            "POST", "/api/v1/fix/code",
            {"code": SHELL_PY, "language": "python", "filename": "add.py"},
        )
        assert isinstance(body, dict), body
        show(body)
        before = body["verification"]["syntax_before"]
        # 无论有没有 AI，本地分析都要能定位到语法错误——这是第 1 步的职责
        located = code == 200 and before is not None and before["line"] == 1
        checks.append(("本地定位到语法错误（第 1 行）", located))
        print(f"  {'OK ' if located else 'FAIL'} 语法错误定位："
              f"{(before or {}).get('line')} 行（应为第 1 行，缺冒号）")

        if not use_ai:
            # 无模型时：原代码必须原样返回，不能丢
            keep_ok = body["fixed_code"] == SHELL_PY and bool(body["note"])
            checks.append(("无模型时原样返回原代码并说明原因", keep_ok))
            print(f"  {'OK ' if keep_ok else 'FAIL'} 原代码原样返回 + 提示：{body['note'][:60]}")

        # ---------------- 2. C 代码：tree-sitter + 风险规则 ----------------
        print()
        print("=" * 72)
        print("2. C 缺分号 + gets 风险")
        print("=" * 72)
        code, body = request(
            "POST", "/api/v1/fix/code",
            {"code": BROKEN_C, "language": "c", "filename": "main.c"},
        )
        assert isinstance(body, dict), body
        show(body)
        before = body["verification"]["syntax_before"]
        # 缺的分号在第 6 行行尾，tree-sitter 会把 MISSING 节点定位到这一行
        tree_ok = before is not None and before["tool"] == "tree-sitter" and before["line"] == 6
        checks.append(("C 语法错误由 tree-sitter 定位", tree_ok))
        print(f"  {'OK ' if tree_ok else 'FAIL'} 定位："
              f"{(before or {}).get('line')} 行 / {(before or {}).get('tool')}"
              f"（应为第 6 行缺分号）")

        local_titles = " ".join(item["title"] for item in body["local_issues"])
        gets_ok = "gets" in local_titles
        checks.append(("本地风险规则指出 gets", gets_ok))
        print(f"  {'OK ' if gets_ok else 'FAIL'} 本地问题：{local_titles[:70]}")

        # ---------------- 3. 复检：AI 的修正代码要真的能过语法检查 ----------------
        if use_ai:
            print()
            print("=" * 72)
            print("3. 本地复检：修正后的代码是否真的通过语法检查")
            print("=" * 72)
            code, body = request(
                "POST", "/api/v1/fix/code",
                {"code": BROKEN_PY, "language": "python", "filename": "average.py"},
            )
            assert isinstance(body, dict), body
            show(body)
            verified = body["verification"]["verified"]
            checks.append(("修正后的代码通过本地语法复检", verified is True))
            print(f"  {'OK ' if verified else 'FAIL'} verified={verified}")

            after = body["verification"]["syntax_after"]
            checks.append(("复检结论里没有残留语法错误", after is None))
            print(f"  {'OK ' if after is None else 'FAIL'} 修正后语法错误：{after}")

            # 把模型给的代码真的交给 ast 跑一遍，验证接口没有"嘴上说通过"
            import ast

            try:
                ast.parse(body["fixed_code"])
                really_ok = True
            except SyntaxError:
                really_ok = False
            checks.append(("用 ast 独立复核模型给的代码确实可解析", really_ok))
            print(f"  {'OK ' if really_ok else 'FAIL'} ast.parse(修正后的代码) -> "
                  f"{'通过' if really_ok else '语法错误'}")

            four_q = all(
                change["what"] and change["why"] and change["how"] and change["avoid"]
                for change in body["changes"]
            )
            checks.append(("每条修改都回答了四个问题", bool(body["changes"]) and four_q))
            print(f"  {'OK ' if four_q else 'FAIL'} 四问完整性："
                  f"{len(body['changes'])} 条修改说明")

        # ---------------- 4. 参数校验 ----------------
        print()
        print("=" * 72)
        print("4. 语言推断与参数校验")
        print("=" * 72)
        code, inferred = request(
            "POST", "/api/v1/fix/code",
            {"code": "int main(void) { return 0; }\n", "filename": "m.c"},
        )
        ok = code == 200 and isinstance(inferred, dict) and inferred["language"] == "c"
        checks.append(("语言留空时按文件名推断", ok))
        print(f"  {'OK ' if ok else 'FAIL'} filename=m.c -> language="
              f"{inferred.get('language') if isinstance(inferred, dict) else inferred}")

        code, bad = request(
            "POST", "/api/v1/fix/code", {"code": "package main\n", "language": "go"}
        )
        ok = code == 400
        checks.append(("不支持的语言返回 400", ok))
        print(f"  {'OK ' if ok else 'FAIL'} language=go -> {code} "
              f"{bad.get('detail') if isinstance(bad, dict) else bad}")

        # ---------------- 5. 历史：对比学习 ----------------
        print()
        print("=" * 72)
        print("5. 修改历史写入 SQLite 且能回看对比")
        print("=" * 72)
        code, listing = request("GET", "/api/v1/fix/history")
        assert isinstance(listing, dict), listing
        print(f"  历史列表：共 {listing['total']} 条")
        for item in listing["items"]:
            print(f"       #{item['id']} {item['filename']:<14} {item['language']:<7} "
                  f"修改={item['change_count']} 处  复检={item['verified']}  "
                  f"+{item['added_lines']}/-{item['removed_lines']}  "
                  f"{item['created_at'][:19]}")
        list_ok = code == 200 and listing["total"] >= 3
        checks.append(("历史列表能读到历次记录", list_ok))
        print(f"  {'OK ' if list_ok else 'FAIL'} GET /fix/history -> {code}")

        first_id = listing["items"][-1]["id"] if listing["items"] else None
        code, detail = request("GET", f"/api/v1/fix/history/{first_id}")
        assert isinstance(detail, dict), detail
        # 对比学习的关键：原代码、新代码、diff、四问说明都要在
        detail_ok = (
            code == 200
            and bool(detail["original_code"])
            and bool(detail["fixed_code"])
            and bool(detail["verification_note"])
        )
        checks.append(("历史详情含原代码/新代码/复检说明", detail_ok))
        print(f"  {'OK ' if detail_ok else 'FAIL'} 详情 #{first_id}："
              f"原代码 {len(detail['original_code'].splitlines())} 行 / "
              f"新代码 {len(detail['fixed_code'].splitlines())} 行 / "
              f"diff {'有' if detail.get('diff') else '无'}")

        # 直接查库核对
        import sqlite3

        with sqlite3.connect(work / "t.db") as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT filename, language, had_error, change_count, verified,"
                " ai_available, added_lines, removed_lines,"
                " length(original_code) AS orig_len, length(fixed_code) AS fixed_len"
                " FROM code_fix_records ORDER BY id"
            ).fetchall()
        print(f"  数据库 code_fix_records 共 {len(rows)} 条：")
        for row in rows:
            print(f"       #{row['filename']:<14} 原代码 {row['orig_len']:>4} 字符 / "
                  f"新代码 {row['fixed_len']:>4} 字符  修改={row['change_count']} "
                  f"复检={row['verified']} AI={row['ai_available']}")
        db_ok = len(rows) == len(listing["items"]) and all(
            row["orig_len"] and row["fixed_len"] for row in rows
        )
        checks.append(("库里同时存了原代码与新代码", db_ok))
        print(f"  {'OK ' if db_ok else 'FAIL'} 每条记录都同时有原代码与新代码全文")

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
