"""纯文本处理的公共小工具。

放在这里的都是「多个模块必须算出同一个结果」的口径，最典型的就是行数：
项目库列表、代码分析结果、历史记录三处都要显示行数，如果各自用
`code.count("\\n") + 1` 这种写法，同一个文件在不同地方会显示成不同的行数，
学生一眼就能看出来不对。所以统一用本模块的函数。
"""

from __future__ import annotations


def count_text_lines(text: str) -> int:
    """按「编辑器口径」统计行数。

    规则（与 VS Code / Notepad++ 左下角显示的一致）：
    - 空文本 -> 0 行；
    - 末尾的换行符不额外算一行：``"a\\nb\\n"`` 是 2 行，不是 3 行；
    - 最后一行没有换行符也算一行：``"a\\nb"`` 是 2 行；
    - 只有换行符的 ``"\\n"`` 是 1 行（一个空行）；
    - CRLF（``\\r\\n``）按一个换行计算，Windows 上的作业不会被算成双倍。
    """
    if not text:
        return 0
    newlines = text.count("\n")
    if newlines == 0:
        return 1
    return newlines if text.endswith("\n") else newlines + 1


__all__ = ["count_text_lines"]
