"""PR / Issue 草稿生成（原 Step 8）。

设计取舍：**模板化生成，不依赖 LLM**。

PR 描述是高度结构化的产物（问题、根因、改动、验证），模板能稳定产出，
而且数据全部来自链路中已经确认过的真实结果（测试计数、覆盖率、diff）。
交给 LLM 自由发挥反而会引入无法核实的表述——在"给开源项目提 PR"这个场景里，
描述与代码不符是比文笔差严重得多的问题。

若要润色措辞，可在此基础上叠加可选的 LLM 改写；本模块保持纯函数、可离线运行。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# 正文里 diff 的展示上限：完整 diff 仍通过接口字段返回
MAX_DIFF_CHARS_IN_BODY = 4000
MAX_BODY_CHARS = 60_000

# 约定式提交前缀
_COMMIT_PREFIX = "fix"


@dataclass(slots=True)
class PrDraft:
    """一份可提交的 PR 草稿。"""

    title: str
    body: str
    branch_name: str
    commit_message: str
    base_branch: str
    files_changed: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0
    is_draft: bool = True
    labels: list[str] = field(default_factory=list)
    # 供调用方复核：这份草稿是否基于一次真正成功的修复
    verified: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "body": self.body,
            "branch_name": self.branch_name,
            "commit_message": self.commit_message,
            "base_branch": self.base_branch,
            "files_changed": self.files_changed,
            "additions": self.additions,
            "deletions": self.deletions,
            "is_draft": self.is_draft,
            "labels": self.labels,
            "verified": self.verified,
            "warnings": self.warnings,
        }


def _slugify(text: str, *, fallback: str = "fix") -> str:
    """把任意文本转成可用的分支名片段。"""
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "-", text or "").strip("-").lower()
    # 分支名只用 ASCII 更稳妥（部分 git 服务对非 ASCII 分支名支持不佳）
    cleaned = re.sub(r"[^0-9a-z-]+", "", cleaned)
    return (cleaned or fallback)[:40].strip("-") or fallback


def _summarize_issue(issue: str, *, limit: int = 60) -> str:
    """从 Issue 描述里取一句做标题素材。"""
    first_line = (issue or "").strip().splitlines()[0] if issue else ""
    return first_line[:limit] if first_line else "自动修复"


def _diff_stats(diff: str) -> tuple[int, int]:
    """统计 diff 的增删行数。"""
    additions = deletions = 0
    for line in diff.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1
    return additions, deletions


def _format_iterations(iterations: list[dict[str, object]]) -> str:
    """把每轮循环渲染成表格行。"""
    lines = ["| 轮次 | 结果 | 失败归类 | 补丁 |", "| --- | --- | --- | --- |"]
    for item in iterations:
        index = item.get("index", "?")
        passed = item.get("passed", 0)
        failed = item.get("failed", 0)
        errors = item.get("errors", 0)
        outcome = f"{passed} 通过 / {failed} 失败 / {errors} 错误"
        category = item.get("category") or "-"
        files = item.get("patch_files") or []
        if item.get("patch_applied"):
            patch = f"已应用（{', '.join(str(f) for f in files)}）"
        else:
            patch = "-"
        lines.append(f"| {index} | {outcome} | {category} | {patch} |")
    return "\n".join(lines)


def build_pr_draft(
    *,
    issue: str,
    repository: str,
    target_path: str,
    target_name: str,
    success: bool,
    status: str,
    message: str,
    diff: str,
    files_changed: list[str],
    iterations: list[dict[str, object]],
    passed: int = 0,
    failed: int = 0,
    errors: int = 0,
    coverage_percent: float | None = None,
    generated_test: str = "",
    run_id: str = "",
    trace_id: str | None = None,
    base_branch: str = "main",
) -> PrDraft:
    """根据自动修复的真实结果生成 PR 草稿。

    只陈述链路中确认过的事实：测试计数、覆盖率、实际改动文件都来自执行结果。
    未成功收敛时如实标注为 draft 并附上失败原因，绝不包装成"修复完成"。
    """
    warnings: list[str] = []
    if not success:
        warnings.append(
            f"自动修复未收敛（status={status}）：{message}。"
            "草稿中如实标注，合并前请人工复核。"
        )
    if not files_changed:
        warnings.append("没有任何文件改动，这份草稿可能不应提交。")
    if failed or errors:
        warnings.append(f"最终执行仍有 {failed} 个失败、{errors} 个错误用例。")

    additions, deletions = _diff_stats(diff)
    summary = _summarize_issue(issue)
    branch = f"{_COMMIT_PREFIX}/agent-{_slugify(summary)}"
    if run_id:
        branch = f"{branch}-{run_id[:6]}"

    title = f"{_COMMIT_PREFIX}: {summary}"
    if not success:
        title = f"[WIP] {title}"

    # 根因来自 Fix Agent 的分析，取第一条非空即可（多轮时以第一轮为准）
    root_cause = ""
    for item in iterations:
        analysis = str(item.get("analysis") or "").strip()
        if analysis:
            root_cause = analysis
            break

    test_note = ""
    if generated_test:
        test_note = (
            "新增的测试由测试生成 Agent 产出，已通过沙箱实际执行验证。"
        )

    diff_block = diff or "(无改动)"
    if len(diff_block) > MAX_DIFF_CHARS_IN_BODY:
        diff_block = (
            diff_block[:MAX_DIFF_CHARS_IN_BODY]
            + "\n...<diff 已截断，完整内容见接口返回的 final_diff 字段>"
        )

    coverage_line = (
        f"- 覆盖率：**{coverage_percent:.1f}%**\n" if coverage_percent is not None else ""
    )
    verification_mark = "✅ 通过" if success else "❌ 未通过"

    body = f"""## 问题

{issue.strip() or "(未提供 Issue 描述)"}

## 根因

{root_cause or "（自动分析未给出明确根因，请人工确认）"}

## 修改内容

- 目标：`{target_path}` 中的 `{target_name}`
- 改动文件：{", ".join(f"`{item}`" for item in files_changed) or "（无）"}
- 变更规模：+{additions} / -{deletions}

## 验证

自动化验证结果：**{verification_mark}**

- 用例：{passed} 通过 / {failed} 失败 / {errors} 错误
{coverage_line}- 修复轮次：{len(iterations)}
{("- " + test_note) if test_note else ""}

### 各轮执行情况

{_format_iterations(iterations)}

## 变更摘要

```diff
{diff_block}
```

---

<sub>本 PR 由「开源项目智能测试与贡献 Agent」自动生成，基于真实执行结果；
修复在隔离工作副本中完成，测试经沙箱实际执行验证。
{'trace_id: `' + trace_id + '`' if trace_id else ''}
仓库: `{repository}` · run_id: `{run_id}`</sub>
"""

    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n\n...<正文已截断>"

    return PrDraft(
        title=title,
        body=body,
        branch_name=branch,
        commit_message=(
            f"{_COMMIT_PREFIX}: {summary}\n\n"
            f"自动修复 {target_path}::{target_name}（run_id={run_id}）"
        ),
        base_branch=base_branch,
        files_changed=list(files_changed),
        additions=additions,
        deletions=deletions,
        is_draft=True,  # 一律作为草稿提交，不直接开正式 PR
        labels=["ai-generated", "needs-review"],
        verified=success and bool(files_changed),
        warnings=warnings,
    )


def build_issue_comment(
    *,
    status: str,
    success: bool,
    message: str,
    target_path: str,
    target_name: str,
    passed: int = 0,
    failed: int = 0,
    errors: int = 0,
    coverage_percent: float | None = None,
    run_id: str = "",
) -> str:
    """生成给 Issue 的评论（用于回报自动修复结果）。"""
    emoji = "✅" if success else "⚠️"
    coverage_line = (
        f"| 覆盖率 | {coverage_percent:.1f}% |\n" if coverage_percent is not None else ""
    )
    return f"""{emoji} **自动修复尝试结果：{status}**

针对 `{target_path}` 中的 `{target_name}`：

| 项目 | 结果 |
| --- | --- |
| 状态 | {status} |
| 用例 | {passed} 通过 / {failed} 失败 / {errors} 错误 |
{coverage_line}| 说明 | {message} |

<sub>run_id: `{run_id}` · 由开源项目智能测试与贡献 Agent 生成</sub>
"""


def utc_timestamp() -> str:
    """当前 UTC 时间戳（便于草稿里标注生成时间）。"""
    return datetime.now(UTC).isoformat()


__all__ = [
    "MAX_BODY_CHARS",
    "MAX_DIFF_CHARS_IN_BODY",
    "PrDraft",
    "build_issue_comment",
    "build_pr_draft",
    "utc_timestamp",
]
