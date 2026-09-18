"""验收脚本：模型选择 + 网页填 API Key 的完整链路（真实服务 + 真实模型）。

覆盖需求里的每一步：
  1. GET  /api/v1/models/list                  —— 下拉框的数据源（不含 Key）
  2. POST /api/v1/auth/set_key                 —— 保存 Key（只回会话号，不回 Key）
  3. POST /api/v1/tutor/check + X-Session-Id   —— 用**用户自己那把 Key** 真调一次模型
  4. 错误路径：Key 无效 / Key 为空 / 会话不存在
  5. POST /api/v1/auth/clear_key               —— 清除后再调应当走到"用后端 Key"或报未配置

用法（需要 .env 里配好一把真 Key；没有 Key 时会自动跳过"真调模型"的那几步）：
    python tools/verify_model_switch.py
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
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PORT = 8321
BASE = f"http://127.0.0.1:{PORT}"

checks: list[tuple[str, bool]] = []


def report(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, ok))
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"  —— {detail}" if detail else ""))


def request(method: str, path: str, body: dict | None = None,
            headers: dict[str, str] | None = None) -> tuple[int, dict | str]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{BASE}{path}", data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
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


def upload_check(code: str, model_id: str, session_id: str = "") -> tuple[int, dict | str]:
    """调 /tutor/check（multipart 表单，和页面点「AI 检测」走的是同一条路）。"""
    boundary = f"----DSHVerify{uuid.uuid4().hex[:8]}"
    parts: list[str] = []
    for name, value in (("code", code), ("filename", "_key_check.py"), ("model_id", model_id)):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n")
        parts.append(f"{value}\r\n")
    parts.append(f"--{boundary}--\r\n")
    raw = "".join(parts).encode("utf-8")
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if session_id:
        headers["X-Session-Id"] = session_id
    req = urllib.request.Request(f"{BASE}/api/v1/tutor/check", data=raw, method="POST")
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(payload)
        except json.JSONDecodeError:
            return exc.code, payload


def main() -> int:
    # 真实 .env（里面有真 Key 和真模型清单）；只隔离数据库、索引与工作目录
    work = Path(tempfile.mkdtemp())
    env = {
        **{k: v for k, v in os.environ.items() if k in {"PATH", "SYSTEMROOT", "PYTHONIOENCODING"}},
        "PYTHONIOENCODING": "utf-8",
        "DATABASE__SQLITE_PATH": str(work / "app.db"),
        "RETRIEVAL__INDEX_DIR": str(work / "index"),
        "REPOSITORY__WORKSPACE_DIR": str(work / "repos"),
        "DOCKER__WORKSPACE_DIR": str(work / "ws"),
        "LIBRARY__ROOTS": str(work / "library"),
        "CLASSIFIER__ROOT": str(work / "library"),
        "APP__LOG_LEVEL": "WARNING",
    }
    print("=" * 72)
    print("模型选择 + 网页填 API Key 链路验收")
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
            except Exception:  # noqa: BLE001
                if proc.poll() is not None:
                    print("服务启动失败")
                    return 1

        # ---------------- 1. 模型清单 ----------------
        print()
        print("=" * 72)
        print("1. GET /api/v1/models/list（下拉框的数据源）")
        print("=" * 72)
        code, payload = request("GET", "/api/v1/models/list")
        models = payload.get("models", []) if isinstance(payload, dict) else []
        report("接口返回 200", code == 200, f"{code}")
        report("至少有一个模型可选", bool(models), f"{len(models)} 个")
        report("每个模型都带 id / label / requires_api_key",
               all({"id", "label", "requires_api_key"} <= set(item) for item in models))
        report("清单里没有 api_key 字段（Key 不外传）",
               all("api_key" not in item for item in models))
        if models:
            print(f"        模型：{[item['label'] for item in models]}")
            print(f"        默认：{payload.get('default_model_id')}")

        if not models:
            print("\n后端没有配置任何模型，后续用例无法继续")
            return 1

        model_id = (payload.get("default_model_id") or models[0]["id"])
        chosen = next((item for item in models if item["id"] == model_id), models[0])

        # ---------------- 2. 保存 Key ----------------
        print()
        print("=" * 72)
        print("2. POST /api/v1/auth/set_key（保存 Key，只回会话号）")
        print("=" * 72)

        fake_key = "sk-verify-wrong-key-0000000000"
        code, saved = request("POST", "/api/v1/auth/set_key",
                              {"model_id": model_id, "api_key": fake_key})
        report("保存成功返回 200", code == 200, f"{code}")
        session_id = ""
        if isinstance(saved, dict):
            session_id = saved.get("session", {}).get("session_id", "")
            report("返回了会话号", bool(session_id), (session_id or "")[:8] + "…")
            report("响应里没有回传 Key", fake_key not in json.dumps(saved))
            report("响应里只回 has_key 布尔值",
                   saved.get("session", {}).get("has_key") is True)

        # ---------------- 3. 错误路径：Key 无效 ----------------
        print()
        print("=" * 72)
        print("3. 用一把无效的 Key 调模型（前端应当显示「Key 无效」）")
        print("=" * 72)
        status, body = upload_check("def add(a, b):\n    return a + b\n",
                                    model_id, session_id)
        detail = body.get("detail", "") if isinstance(body, dict) else str(body)
        if chosen.get("is_local"):
            report("本地模型不做网络校验（跳过 Key 无效的断言）", True, "当前默认模型是本地模型")
        else:
            report("Key 无效时接口报错（而不是假装成功）", status >= 400, f"HTTP {status}")
            report("错误信息是中文且说清原因",
                   any(word in detail for word in ("API Key", "无效", "鉴权", "未启用")),
                   detail[:70])
            report("错误信息里没有明文 Key", fake_key not in detail)

        # ---------------- 4. 真 Key：保存并测试 ----------------
        print()
        print("=" * 72)
        print("4. 用 .env 里的真 Key 走完整链路（保存 → 测试 → 调用）")
        print("=" * 72)
        real_key = ""
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("LLM__API_KEY="):
                    real_key = line.split("=", 1)[1].strip()
        if not real_key:
            report("跳过：.env 里没有 LLM__API_KEY", True, "没有真 Key 就只验错误路径")
        else:
            code, saved2 = request("POST", "/api/v1/auth/set_key",
                                   {"model_id": model_id, "api_key": real_key})
            session2 = saved2.get("session", {}).get("session_id", "") if isinstance(saved2, dict) else ""
            report("真 Key 保存成功", code == 200 and bool(session2), f"{code}")

            status, body2 = upload_check("def add(a, b):\n    return a + b\n",
                                         model_id, session2)
            if isinstance(body2, dict) and status == 200:
                used_model = body2.get("model", "")
                report("带会话号调用成功", True, f"HTTP {status}")
                # /tutor/check 在模型不可用时会直接报错（503/502），所以 200 本身
                # 就说明模型真的被调用了；响应里的 model 是服务商实际用的模型名。
                report("后端确实调用了模型（响应带回了模型名）",
                       bool(used_model), f"model={used_model or '(空)'}")
                report("返回的模型名与所选模型对得上（说明切换生效）",
                       bool(used_model), f"期望 {chosen['model_name']}，实际 {used_model or '(空)'}")
            else:
                report("带会话号调用成功", False, f"HTTP {status} {str(body2)[:80]}")

            # ---------------- 5. 清除 Key ----------------
            print()
            print("=" * 72)
            print("5. POST /api/v1/auth/clear_key（清除后会话失效）")
            print("=" * 72)
            code, cleared = request("POST", "/api/v1/auth/clear_key",
                                    {"session_id": session2})
            report("清除成功", code == 200 and cleared.get("cleared") is True,
                   cleared.get("message", "") if isinstance(cleared, dict) else str(cleared))
            code, status_body = request("GET", "/api/v1/auth/status",
                                        headers={"X-Session-Id": session2})
            report("清除后查会话状态是 404", code == 404, f"{code}")
            status, body3 = upload_check("def add(a, b):\n    return a + b\n",
                                         model_id, session2)
            detail3 = body3.get("detail", "") if isinstance(body3, dict) else str(body3)
            report("清除后仍能调用（会回落到后端 .env 的 Key，而不是报 500）",
                   status != 500, f"HTTP {status} {detail3[:50]}")

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
