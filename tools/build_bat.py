"""把 tools/bat_src 下的 .bat.utf8 源文件编译成项目根目录可直接双击的 .bat。

用法：
    python tools/build_bat.py

两个必须同时满足的 Windows 批处理文件要求：

1. 换行符必须是 CRLF（\\r\\n）。
   用裸 LF 时 cmd.exe 的解析器会错乱：丢掉 REM 前缀、把多行粘成一行，
   报出大量「不是内部或外部命令」。实测这是最容易忽略的坑 ——
   用现代编辑器（默认 LF）保存 .bat 后直接运行就会踩到。

2. 编码必须与 cmd 读取时的控制台代码页一致。
   简体中文 Windows 默认 936(GBK)，因此文件存成 GBK，
   并在 @echo off 之后立刻 chcp 936，且该行上方不得出现任何中文
   （cmd 会用「执行到 chcp 之前那一刻的代码页」去解析文件内容）。

因此项目保留 UTF-8 + LF 的 .utf8 源文件便于用现代编辑器维护，
由本脚本负责生成可双击运行的最终文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC_DIR = BASE / "tools" / "bat_src"
PAIRS = [("start.bat.utf8", "start.bat"), ("stop.bat.utf8", "stop.bat")]

failures = 0
for src_name, dst_name in PAIRS:
    src = SRC_DIR / src_name
    dst = BASE / dst_name
    text = src.read_text(encoding="utf-8")

    problems: list[str] = []

    # --- 1) 统一成 CRLF ---
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    crlf_text = "\r\n".join(lines)

    # --- 2) 校验：chcp 行之前不得有非 ASCII ---
    chcp_index = next(
        (i for i, ln in enumerate(lines) if ln.strip().lower().startswith("chcp ")), None
    )
    if chcp_index is None:
        problems.append("没有找到 chcp 行")
    else:
        for i in range(chcp_index):
            if any(ord(ch) > 127 for ch in lines[i]):
                problems.append(
                    f"第 {i + 1} 行在 chcp 之前含非 ASCII：{lines[i][:40]}"
                )

    # --- 3) 转 GBK ---
    try:
        payload = crlf_text.encode("gbk")
    except UnicodeEncodeError as exc:
        problems.append(f"含 GBK 无法表示的字符：{exc}")
        print(f"[X] {dst_name}: {problems}")
        continue

    dst.write_bytes(payload)

    # --- 4) 回读校验 ---
    raw = dst.read_bytes()
    roundtrip_ok = raw.decode("gbk") == crlf_text
    crlf_count = raw.count(b"\r\n")
    bare_lf = raw.count(b"\n") - crlf_count

    if bare_lf:
        problems.append(f"仍存在 {bare_lf} 个裸 LF")
    if crlf_count != len(lines) - 1 and not crlf_text.endswith("\r\n"):
        problems.append(f"CRLF 数量异常：{crlf_count}")

    status = "OK" if not problems else "需修正"
    if problems:
        failures += 1
    print(f"[{status}] {src_name} -> {dst_name}")
    print(f"        编码 GBK / 换行 CRLF({crlf_count} 个) / 裸 LF({bare_lf} 个)")
    print(f"        {len(crlf_text)} 字符 -> {len(raw)} 字节, 回读一致={roundtrip_ok}")
    print(f"        chcp 位于第 {chcp_index + 1 if chcp_index is not None else '?'} 行")
    for item in problems:
        print(f"        [!] {item}")

if failures:
    print(f"\n有 {failures} 个文件未通过校验，请修正源文件后重试。")
    sys.exit(1)
print("\n全部通过：生成的 .bat 可直接双击运行。")
