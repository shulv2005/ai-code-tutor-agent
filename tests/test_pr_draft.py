"""PR 草稿生成测试。"""

from __future__ import annotations

from app.services.pr_draft import (
    MAX_BODY_CHARS,
    MAX_DIFF_CHARS_IN_BODY,
    build_issue_comment,
    build_pr_draft,
)

DIFF = """\
diff --git a/calc.py b/calc.py
--- a/calc.py
+++ b/calc.py
@@ -1,3 +1,3 @@
 def add(a, b):
     \"\"\"Add.\"\"\"
-    return a - b
+    return a + b
"""

ITERATIONS = [
    {
        "index": 1,
        "passed": 0,
        "failed": 2,
        "errors": 0,
        "category": "source_bug",
        "analysis": "实现用了减号，测试期望求和。",
        "patch_files": ["calc.py"],
        "patch_applied": True,
    },
    {"index": 2, "passed": 2, "failed": 0, "errors": 0, "category": None, "patch_applied": False},
]


def _draft(**overrides: object):
    base: dict[str, object] = {
        "issue": "add 两个数相加结果是错的",
        "repository": "e2e/demo",
        "target_path": "calc.py",
        "target_name": "add",
        "success": True,
        "status": "passed",
        "message": "测试通过，无需修复",
        "diff": DIFF,
        "files_changed": ["calc.py", "agent_tests/test_generated.py"],
        "iterations": ITERATIONS,
        "passed": 2,
        "failed": 0,
        "errors": 0,
        "coverage_percent": 100.0,
        "generated_test": "def test_add(): ...",
        "run_id": "abc123def456",
        "trace_id": "trace-xyz",
        "base_branch": "main",
    }
    base.update(overrides)
    return build_pr_draft(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 成功场景
# ---------------------------------------------------------------------------
def test_successful_draft_metadata() -> None:
    draft = _draft()

    assert draft.title.startswith("fix:")
    assert not draft.title.startswith("[WIP]")
    assert draft.branch_name.startswith("fix/agent-")
    assert draft.base_branch == "main"
    assert draft.is_draft is True
    assert draft.verified is True
    assert "ai-generated" in draft.labels
    assert draft.warnings == []


def test_diff_stats_are_counted() -> None:
    draft = _draft()
    assert draft.additions == 1
    assert draft.deletions == 1


def test_body_contains_required_sections() -> None:
    body = _draft().body
    for section in ("## 问题", "## 根因", "## 修改内容", "## 验证", "## 变更摘要"):
        assert section in body


def test_body_includes_real_facts() -> None:
    """正文只应陈述链路中确认过的事实。"""
    body = _draft().body
    assert "add 两个数相加结果是错的" in body
    assert "实现用了减号" in body  # 根因来自 Fix Agent
    assert "2 通过 / 0 失败 / 0 错误" in body
    assert "100.0%" in body
    assert "calc.py" in body
    assert "trace-xyz" in body


def test_body_includes_iteration_table() -> None:
    body = _draft().body
    assert "| 轮次 |" in body
    assert "source_bug" in body


# ---------------------------------------------------------------------------
# 未收敛场景：必须如实标注
# ---------------------------------------------------------------------------
def test_unsuccessful_draft_is_marked_wip() -> None:
    draft = _draft(
        success=False,
        status="max_attempts",
        message="已达最大修复轮次（3），测试仍未通过",
        failed=2,
    )
    assert draft.title.startswith("[WIP]")
    assert draft.verified is False
    assert draft.warnings
    assert any("未收敛" in item for item in draft.warnings)
    assert any("3" in item for item in draft.warnings)


def test_draft_without_changes_warns() -> None:
    draft = _draft(files_changed=[], diff="", success=False, status="no_patch")
    assert any("没有任何文件改动" in item for item in draft.warnings)


def test_missing_root_cause_is_stated() -> None:
    draft = _draft(iterations=[{"index": 1, "passed": 1, "failed": 0}])
    assert "未给出明确根因" in draft.body


def test_missing_issue_description_is_handled() -> None:
    draft = _draft(issue="")
    assert "未提供 Issue 描述" in draft.body
    assert draft.title


# ---------------------------------------------------------------------------
# 边界
# ---------------------------------------------------------------------------
def test_large_diff_is_truncated_in_body() -> None:
    huge = "--- a/x.py\n+++ b/x.py\n" + "".join(f"+line {i}\n" for i in range(2000))
    draft = _draft(diff=huge)
    assert len(draft.body) <= MAX_BODY_CHARS
    assert "diff 已截断" in draft.body
    assert len(draft.body.split("```diff")[1]) < MAX_DIFF_CHARS_IN_BODY + 500


def test_chinese_issue_produces_ascii_branch() -> None:
    draft = _draft(issue="取消订单时没有校验状态")
    # 分支名只用 ASCII，避免部分 git 服务不支持非 ASCII 分支名
    assert draft.branch_name.isascii()
    assert " " not in draft.branch_name
    assert ".." not in draft.branch_name


def test_branch_name_is_slug_safe_for_symbols_only_issue() -> None:
    draft = _draft(issue="!!!")
    assert draft.branch_name.startswith("fix/agent-")
    assert draft.branch_name.isascii()


def test_run_id_suffix_makes_branch_unique() -> None:
    a = _draft(run_id="aaaaaa111111")
    b = _draft(run_id="bbbbbb222222")
    assert a.branch_name != b.branch_name


def test_commit_message_has_conventional_prefix() -> None:
    draft = _draft()
    assert draft.commit_message.startswith("fix:")
    assert "calc.py::add" in draft.commit_message


# ---------------------------------------------------------------------------
# Issue 评论
# ---------------------------------------------------------------------------
def test_issue_comment_reports_result() -> None:
    comment = build_issue_comment(
        status="passed",
        success=True,
        message="测试通过，无需修复",
        target_path="calc.py",
        target_name="add",
        passed=2,
        failed=0,
        coverage_percent=100.0,
        run_id="abc123",
    )
    assert "✅" in comment
    assert "passed" in comment
    assert "calc.py" in comment
    assert "100.0%" in comment


def test_issue_comment_for_failure_uses_warning_mark() -> None:
    comment = build_issue_comment(
        status="no_patch",
        success=False,
        message="模型未能给出可用补丁",
        target_path="x.py",
        target_name="f",
    )
    assert "⚠️" in comment
    assert "no_patch" in comment


def test_to_dict_is_serializable() -> None:
    import json

    payload = json.dumps(_draft().to_dict(), ensure_ascii=False)
    assert "branch_name" in payload
