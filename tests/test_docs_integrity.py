"""多智能体学习笔记仓库文档与 SVG 完整性自动化测试套件。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_docs.py"


def _load_validator_module() -> ModuleType:
    """通过标准库 importlib 加载 scripts/validate_docs.py，避免硬编码 sys.path 污染。"""
    spec = importlib.util.spec_from_file_location("validate_docs", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vd = _load_validator_module()


def _format_issues(issues: list[object]) -> str:
    """格式化失败列表，便于 pytest 报错时精确定位问题。"""
    return "\n".join(f"  - {issue.format()}" for issue in issues)


def test_github_slug_rules() -> None:
    """验证 GitHub 标题锚点转换逻辑（含中文标点、双连字符与去重计数）。"""
    assert (
        vd.github_slug("第一章：第一性原理与信息论边界（为什么单智能体会撞墙？）")
        == "第一章第一性原理与信息论边界为什么单智能体会撞墙"
    )
    assert (
        vd.github_slug("第七章：2025 至 2026 四大主流生产框架运行时与双层开放协议栈（MCP + A2A）")
        == "第七章2025-至-2026-四大主流生产框架运行时与双层开放协议栈mcp--a2a"
    )
    assert (
        vd.github_slug("第八章：智能体评测科学（从静态刷榜到 pass^k 可靠性悬崖与 Agent-as-a-Judge）")
        == "第八章智能体评测科学从静态刷榜到-passk-可靠性悬崖与-agent-as-a-judge"
    )
    seen: dict[str, int] = {}
    assert vd.github_slug("小结", seen) == "小结"
    assert vd.github_slug("小结", seen) == "小结-1"


def test_markdown_links_anchors_and_no_orphan_assets() -> None:
    """所有本地相对链接、图片路径与标题锚点均应有效，且无孤立资源或孤立笔记。"""
    issues, checked_links, checked_anchors = vd.validate_links_and_assets(REPO_ROOT)
    assert checked_links >= 20, "预期至少校验 20 处以上本地链接或图片引用"
    assert checked_anchors >= 15, "预期至少校验 15 处以上页内或跨文件锚点"
    assert not issues, f"发现链接/锚点/孤立资源问题：\n{_format_issues(issues)}"


def test_svg_xml_attributes_and_layout_bounds() -> None:
    """所有 11 张 SVG 均为严格合法 XML，包含 xmlns/viewBox，且无文本越界。"""
    issues, checked_svgs = vd.validate_svgs(REPO_ROOT)
    assert checked_svgs == 11, f"预期 assets/ 下有 11 张原创 SVG，实际为 {checked_svgs}"
    assert not issues, f"发现 SVG 完整性或排版越界问题：\n{_format_issues(issues)}"


def test_figure_numbering_and_toc_consistency() -> None:
    """主书图号严格按 1..11 顺序递增，SVG 图号与 Markdown 图注对齐，目录与章节标题一致。"""
    issues = vd.validate_figure_numbering_and_toc(REPO_ROOT)
    assert not issues, f"发现目录或图号一致性问题：\n{_format_issues(issues)}"


def test_markdown_tables_fences_and_latex_balance() -> None:
    """所有 Markdown 代码块带语言标识、表格列数对齐、LaTeX 公式成对闭合。"""
    issues = vd.validate_markdown_and_latex(REPO_ROOT)
    assert not issues, f"发现 Markdown/LaTeX 语法问题：\n{_format_issues(issues)}"


def test_no_sensitive_information_leaked() -> None:
    """仓库全部文档与 SVG 不得包含真实密钥、私网 IP 或未脱敏内部地址。"""
    issues = vd.validate_sensitive_info(REPO_ROOT)
    assert not issues, f"检测到疑似未脱敏敏感信息：\n{_format_issues(issues)}"


def test_cli_entrypoint_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """验证独立脚本 CLI 端到端返回 0 退出码并输出中文摘要。"""
    exit_code = vd.main(["--root", str(REPO_ROOT)])
    captured = capsys.readouterr()
    assert exit_code == 0, f"CLI 返回非零退出码:\n{captured.out}"
    assert "100% 通过" in captured.out


def test_validator_catches_broken_inputs(tmp_path: Path) -> None:
    """反向防回归测试：确认质检器能够准确捕获断链、坏 SVG、未闭合公式与密钥泄露。"""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "broken.svg").write_text("<svg><text>unclosed", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "# 示例标题\n\n"
        "[坏链接](missing.md)\n\n"
        "[坏锚点](#不存在的标题)\n\n"
        "$$\nx = 1\n\n"
        "内网测试地址：192.168.1.42\n",
        encoding="utf-8",
    )

    report = vd.run_all_validations(tmp_path)
    categories = {issue.category for issue in report.issues}
    assert not report.passed
    assert "link" in categories
    assert "anchor" in categories
    assert "svg-xml" in categories
    assert "latex-display" in categories
    assert "sensitive-info" in categories
