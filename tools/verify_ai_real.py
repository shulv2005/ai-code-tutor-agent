"""真实大模型验收：用项目自带的示例作业，实测三个 AI 功能的效果。

与其它验收脚本的区别（很重要）：
- `tests/` 里的用例全部用「假 LLM 客户端」，验证的是**代码路径**正确，
  不能证明模型真的能找出错误、真的会写中文注释；
- 本脚本**不注入任何假客户端**，直接读 .env 里的真实配置，
  对着真实模型跑一遍，把模型的原始结论打印出来给人看。

因此它需要网络与可用的 API Key，属于「人工验收」脚本，不进 pytest。
    运行：python tools/verify_ai_real.py
    前提：.env 里配好 LLM__API_KEY（没配的话本脚本会直接说明并退出）

判定标准（刻意做成客观、可自动判定的，而不是"看着像对的"）：
    检测：分数在 0-100，且至少报出 1 个问题；示例文件里预设的 bug 能被指出（软判定，仅提示）
    注释：输出与原文不同、包含中文、且原有代码行没有被删改（硬判定）
    改错：输出与原文不同、给出了修改条目；示例文件里预设的 bug 被改掉（软判定，仅提示）
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
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PORT = 8312
BASE = f"http://127.0.0.1:{PORT}"

# 待验收的示例作业：文件 + 我们**事先知道**的正确结论，用来判断模型有没有找对
CASES = [
    {
        "path": "examples/示例作业/bubble_sort.c",
        "language": "c",
        "knows": "print_array 循环多跑一轮（数组越界）；比较符号写反导致从大到小",
        # 关键词命中任意一个就算模型指出了这个 bug（不同模型措辞差别很大）
        "bug1": ["越界", "超出", "多一", "off-by-one", "n - 1", "n-1", "<=", "i < n"],
        "bug2": ["小于号", "比较符号", "写反", "reverse", "降序", "从大到小", ">"],
    },
    {
        "path": "examples/示例作业/score_stats.py",
        "language": "python",
        "knows": "平均分分母写成 len(scores) + 1；空列表时取 scores[0] 会抛异常",
        "bug1": ["分母", "len(scores) + 1", "多除了", "除以", "+ 1", "+1", "偏小"],
        "bug2": ["空列表", "IndexError", "越界", "scores[0]", "为空", "长度"],
    },
    {
        "path": "examples/示例作业/ScoreManager.java",
        "language": "java",
        "knows": "getMax 初始值取 0，全是负数时结果错误",
        "bug1": ["初始值", "max = 0", "0", "负数", "Integer.MIN_VALUE", "scores[0]"],
        "bug2": [],
    },
]

CJK = re.compile(r"[\u4e00-\u9fff]")


def request(method: str, path: str, *, raw: bytes | None = None,
            content_type: str | None = None, timeout: int = 180):
    """发一个 HTTP 请求，返回 (状态码, 解析后的内容)。"""
    req = urllib.request.Request(f"{BASE}{path}", data=raw, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def form_body(filename: str, code: str) -> tuple[bytes, str]:
    """手工拼 multipart/form-data 请求体（不引入额外依赖）。"""
    boundary = "----DSHRealAIBoundary"
    parts = []
    for name, value in (("filename", filename), ("code", code)):
        parts.append(f"--{boundary}\r\n")
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        parts.append(f"{value}\r\n")
    parts.append(f"--{boundary}--\r\n")
    return "".join(parts).encode("utf-8"), f"multipart/form-data; boundary={boundary}"


def code_lines_of(text: str) -> list[str]:
    """取出"有实际内容、且不是纯注释"的行，用于判断模型有没有擅自删改代码。"""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("#", "//", "/*", "*", "*/", "'''", '"""')):
            continue
        lines.append(line)
    return lines


def hits(text: str, keywords: list[str]) -> bool:
    """text 里是否出现任一关键词（大小写不敏感）。"""
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def main() -> int:
    # ---- 0. 先确认真的配了 Key，否则整个脚本没有意义 ----
    sys.path.insert(0, str(BASE_DIR))
    from app.core.config import get_settings

    settings = get_settings()
    if not settings.llm.is_configured:
        print("=" * 72)
        print("未配置 LLM__API_KEY：本脚本需要真实模型才能验收，已退出。")
        print("请在 .env 中填好 Key（可参考 .env.example 的说明）后重跑。")
        print("=" * 72)
        return 1
    print("=" * 72)
    print("真实大模型验收（不使用任何假客户端）")
    print(f"端点：{settings.llm.base_url}")
    print(f"模型：{settings.llm.model}")
    print("=" * 72)

    work = Path(tempfile.mkdtemp())
    env = dict(os.environ)
    env.update({
        # 刻意**不覆盖** LLM__* ：本脚本就是要用真实配置
        "DATABASE__SQLITE_PATH": str(work / "t.db"),
        "RETRIEVAL__INDEX_DIR": str(work / "index"),
        "REPOSITORY__WORKSPACE_DIR": str(work / "repos"),
        "DOCKER__WORKSPACE_DIR": str(work / "ws"),
        "LIBRARY__ROOTS": str(BASE_DIR / "examples"),
        "APP__LOG_LEVEL": "WARNING",
        "PYTHONIOENCODING": "utf-8",
    })
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT),
         "--log-level", "warning"],
        cwd=str(BASE_DIR), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    checks: list[tuple[str, bool]] = []
    notes: list[str] = []
    try:
        for _ in range(60):
            time.sleep(0.5)
            try:
                request("GET", "/api/v1/health", timeout=5)
                break
            except Exception:
                if proc.poll() is not None:
                    print("服务启动失败")
                    return 1

        code, status = request("GET", "/api/v1/tutor/status")
        print(f"\n服务状态：{code}  ai_available={status.get('ai_available')} "
              f"model={status.get('model')}")
        if not status.get("ai_available"):
            print("服务报告 AI 不可用，后续验收没有意义，已退出。")
            return 1

        for case in CASES:
            source = (BASE_DIR / case["path"]).read_text(encoding="utf-8")
            filename = Path(case["path"]).name
            print("\n" + "=" * 72)
            print(f"文件：{case['path']}（{len(source.splitlines())} 行，{case['language']}）")
            print(f"预设的错：{case['knows']}")
            print("=" * 72)

            # ---------------- AI 检测 ----------------
            body, ctype = form_body(filename, source)
            started = time.time()
            code, data = request("POST", "/api/v1/tutor/check", raw=body, content_type=ctype)
            elapsed = time.time() - started
            if code != 200 or not isinstance(data, dict):
                checks.append((f"{filename} AI 检测", False))
                print(f"  FAIL 检测失败 HTTP {code}：{str(data)[:200]}")
                continue

            issues = data.get("issues", [])
            all_text = " ".join(
                f"{i.get('title','')} {i.get('detail','')} {i.get('suggestion','')}"
                for i in issues
            )
            found1 = hits(all_text, case["bug1"])
            found2 = hits(all_text, case["bug2"]) if case["bug2"] else None

            ok_check = 0 <= data.get("score", -1) <= 100 and len(issues) >= 1
            checks.append((f"{filename} AI 检测", ok_check))
            print(f"  {'OK ' if ok_check else 'FAIL'} 检测完成 {elapsed:.1f}s  "
                  f"评分={data.get('score')} 等级={data.get('level')} 问题={len(issues)} 条")
            print(f"       总评：{data.get('summary', '')[:100]}")
            for item in issues[:4]:
                print(f"       · [{(item.get('severity') or '').upper()}] "
                      f"L{item.get('line')} {item.get('title', '')[:60]}")
            print(f"       预设 bug1 被指出：{'是' if found1 else '否'}"
                  + ("" if found2 is None else f"  bug2 被指出：{'是' if found2 else '否'}"))
            if not found1:
                notes.append(f"{filename}：模型没有明确提到预设 bug1，建议人工看一眼")
            if found2 is False:
                notes.append(f"{filename}：模型没有明确提到预设 bug2，建议人工看一眼")

            # ---------------- 生成注释 ----------------
            body, ctype = form_body(filename, source)
            started = time.time()
            code, data = request("POST", "/api/v1/tutor/comment", raw=body, content_type=ctype)
            elapsed = time.time() - started
            if code != 200 or not isinstance(data, dict):
                checks.append((f"{filename} 生成注释", False))
                print(f"  FAIL 注释失败 HTTP {code}：{str(data)[:200]}")
                continue

            commented = data.get("commented_code", "")
            cjk_count = len(CJK.findall(commented))
            origin_lines = code_lines_of(source)
            kept = sum(1 for line in origin_lines if line in commented)
            keep_ratio = kept / len(origin_lines) if origin_lines else 1.0

            # 硬判定：有中文注释、代码没被大改（原文代码行保留率 >= 90%）
            ok_comment = (
                commented != source and cjk_count >= 5 and keep_ratio >= 0.9
            )
            checks.append((f"{filename} 生成注释", ok_comment))
            print(f"  {'OK ' if ok_comment else 'FAIL'} 注释完成 {elapsed:.1f}s  "
                  f"输出 {len(commented.splitlines())} 行 / 中文 {cjk_count} 字 / "
                  f"原文代码行保留 {kept}/{len(origin_lines)} ({keep_ratio:.0%})")
            print(f"       概述：{data.get('summary', '')[:100]}")
            if keep_ratio < 0.9:
                notes.append(f"{filename}：注释结果改动了原代码（保留率 {keep_ratio:.0%}），需人工确认")

            # ---------------- 自动改错 ----------------
            body, ctype = form_body(filename, source)
            started = time.time()
            code, data = request("POST", "/api/v1/tutor/fix", raw=body, content_type=ctype)
            elapsed = time.time() - started
            if code != 200 or not isinstance(data, dict):
                checks.append((f"{filename} 自动改错", False))
                print(f"  FAIL 改错失败 HTTP {code}：{str(data)[:200]}")
                continue

            fixed = data.get("fixed_code", "")
            changes = data.get("changes", [])
            change_text = " ".join(
                f"{c.get('original','')} {c.get('fixed','')} {c.get('reason','')}"
                for c in changes
            )
            ok_fix = fixed != source and len(changes) >= 1
            checks.append((f"{filename} 自动改错", ok_fix))
            print(f"  {'OK ' if ok_fix else 'FAIL'} 改错完成 {elapsed:.1f}s  "
                  f"had_error={data.get('had_error')} 修改 {len(changes)} 处")
            print(f"       概述：{data.get('summary', '')[:100]}")
            for change in changes[:4]:
                print(f"       · L{change.get('line')} "
                      f"{str(change.get('original'))[:40]!r} -> {str(change.get('fixed'))[:40]!r}")
                print(f"         理由：{str(change.get('reason'))[:80]}")
            fix1 = hits(change_text + fixed, case["bug1"])
            fix2 = hits(change_text + fixed, case["bug2"]) if case["bug2"] else None
            print(f"       预设 bug1 被修改：{'是' if fix1 else '否'}"
                  + ("" if fix2 is None else f"  bug2 被修改：{'是' if fix2 else '否'}"))

        # ---------------- 历史记录 ----------------
        code, history = request("GET", "/api/v1/tutor/history?limit=50")
        total = history.get("total", 0) if isinstance(history, dict) else 0
        # 3 个文件 × 3 个动作 = 9 条
        ok_history = code == 200 and total >= 9
        checks.append(("历史记录写入 9 条", ok_history))
        print(f"\n{'OK ' if ok_history else 'FAIL'} 历史记录：{code}  共 {total} 条")

        print()
        print("=" * 72)
        passed = sum(1 for _, ok in checks if ok)
        print(f"真实模型验收结果：{passed}/{len(checks)} 项通过")
        print("=" * 72)
        for label, ok in checks:
            print(f"  {'OK  ' if ok else 'FAIL'} {label}")
        if notes:
            print("\n需要人工确认的点：")
            for note in notes:
                print(f"  · {note}")
        return 0 if passed == len(checks) else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
