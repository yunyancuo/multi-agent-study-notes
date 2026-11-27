"""仓库文档与 SVG 资源自动化质量校验脚本（零第三方依赖）。

校验范围：
1. 所有 Markdown 本地相对链接、图片路径与跨文件/页内标题锚点 100% 有效，且无孤立资源；
2. 所有 SVG 文件为严格合法 XML，包含规范 xmlns 与 viewBox，且无明显文本越界或编码损坏；
3. 主书目录导航与正文标题一致，图号严格递增且与 SVG 内部标题编号一致，专题笔记具备返回导航；
4. Markdown 围栏代码块语言标识、表格列数对齐，以及 LaTeX 块级/行内公式成对闭合；
5. 全仓库文本无未脱敏密钥、私网 IP 或真实内网敏感信息泄露。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
ASSET_EXTENSIONS = {".svg", ".png", ".jpg", ".jpeg", ".webp", ".gif"}

# Markdown 与 LaTeX 解析正则
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
LINK_RE = re.compile(r"(!?)\[([^\]]*)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
FIG_NUM_RE = re.compile(r"图\s*(\d+)")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")

# 潜在敏感信息正则清单（类别名称, 正则表达式）
SENSITIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("私钥块", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("OpenAI/通用 sk- 密钥", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("GitHub 个人访问令牌", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("AWS Access Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("Google API Key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("Slack Token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
    (
        "私网 IPv4 地址",
        re.compile(
            r"(?<![\d.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
            r"|172\.(?:1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}"
            r"|192\.168\.\d{1,3}\.\d{1,3})(?![\d.])"
        ),
    ),
    (
        "疑似内部主机域名",
        re.compile(
            r"https?://[A-Za-z0-9.-]+\.(?:internal|corp|lan|local)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "疑似硬编码明文凭据",
        re.compile(
            r"""(?i)\b(?:api_key|secret_key|access_token|password)\s*[:=]\s*["']"""
            r"""(?!\s*(?:your_|example|placeholder|changeme|dummy|test|<|\$|\*))[A-Za-z0-9_\-./+=]{12,}["']"""
        ),
    ),
)


@dataclass(frozen=True)
class ValidationIssue:
    """单条质检问题记录。"""

    category: str
    path: str
    line: int | None
    message: str

    def format(self) -> str:
        """格式化输出人类可读的告警信息。"""
        loc = f"{self.path}:{self.line}" if self.line is not None else self.path
        return f"[{self.category}] {loc} -> {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    """全量文档质检汇总报告。"""

    issues: list[ValidationIssue]
    checked_md_files: int
    checked_svg_files: int
    checked_links: int
    checked_anchors: int

    @property
    def passed(self) -> bool:
        """无任何错误即视为通过。"""
        return len(self.issues) == 0


def strip_markdown_inline(text: str) -> str:
    """移除标题中的 Markdown 内联修饰标记，保留纯展示文本。"""
    cleaned = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"(\*\*|__|\*|_)", "", cleaned)
    return cleaned.strip()


def github_slug(heading: str, seen: dict[str, int] | None = None) -> str:
    """按照 GitHub github-slugger 规则将标题转换为页内锚点标识。"""
    plain = strip_markdown_inline(heading).lower()
    chars: list[str] = []
    for ch in plain:
        if ch == " ":
            chars.append("-")
        elif ch in {"-", "_"}:
            chars.append(ch)
        else:
            cat = unicodedata.category(ch)
            # 保留所有 Unicode 字母（含中文）、数字与组合标记，其余标点符号剔除
            if cat and cat[0] in {"L", "N", "M"}:
                chars.append(ch)
    base_slug = "".join(chars)
    if seen is None:
        return base_slug
    count = seen.get(base_slug, 0)
    seen[base_slug] = count + 1
    return base_slug if count == 0 else f"{base_slug}-{count}"


def iter_non_code_lines(text: str) -> Iterator[tuple[int, str]]:
    """遍历 Markdown 中不在围栏代码块内的行（返回 1-based 行号与文本）。"""
    in_fence = False
    fence_char = ""
    fence_len = 0
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        match = FENCE_RE.match(raw_line)
        if match:
            marker = match.group(1)
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_len and not match.group(2).strip():
                in_fence = False
            continue
        if not in_fence:
            yield line_no, raw_line


def extract_headings(text: str) -> list[tuple[int, int, str, str]]:
    """提取 Markdown 文档全部标题，返回 (行号, 级别, 原标题, GitHub锚点)。"""
    seen: dict[str, int] = {}
    headings: list[tuple[int, int, str, str]] = []
    for line_no, line in iter_non_code_lines(text):
        match = HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            raw_title = match.group(2).strip()
            slug = github_slug(raw_title, seen)
            headings.append((line_no, level, raw_title, slug))
    return headings


def parse_link_target(raw_target: str) -> str:
    """清理 Markdown 链接目标中的可选标题和尖括号包裹。"""
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")].strip()
    # 处理形如 path/to/file.md "title" 的写法
    parts = target.split(maxsplit=1)
    return parts[0] if parts else ""


def extract_markdown_links(text: str) -> list[tuple[int, bool, str, str]]:
    """提取非代码块与非行内代码中的 Markdown 链接与图片引用。

    返回列表元素为 (行号, 是否为图片, 标签文本, 链接目标)。
    """
    results: list[tuple[int, bool, str, str]] = []
    for line_no, line in iter_non_code_lines(text):
        # 将行内代码替换为等长空格占位，避免误匹配示例代码中的方括号链接
        masked = INLINE_CODE_RE.sub(lambda m: " " * len(m.group(0)), line)
        for match in LINK_RE.finditer(masked):
            is_image = match.group(1) == "!"
            label = match.group(2)
            target = parse_link_target(match.group(3))
            if target:
                results.append((line_no, is_image, label, target))
    return results


def discover_markdown_files(repo_root: Path) -> list[Path]:
    """发现仓库中所有待校验的 Markdown 文件（排除隐藏目录与构建缓存）。"""
    excluded = {".git", ".pytest_cache", ".ruff_cache", ".venv", "venv", "__pycache__"}
    files: list[Path] = []
    for path in sorted(repo_root.rglob("*.md")):
        if any(part in excluded for part in path.relative_to(repo_root).parts):
            continue
        files.append(path)
    return files


def validate_links_and_assets(repo_root: Path) -> tuple[list[ValidationIssue], int, int]:
    """校验 Markdown 本地路径、页内/跨文件锚点及孤立资源。"""
    issues: list[ValidationIssue] = []
    md_files = discover_markdown_files(repo_root)
    headings_by_file: dict[Path, set[str]] = {}
    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8")
        headings_by_file[md_path.resolve()] = {slug for _, _, _, slug in extract_headings(text)}

    referenced_files: set[Path] = set()
    checked_links = 0
    checked_anchors = 0

    for md_path in md_files:
        rel_md = md_path.relative_to(repo_root).as_posix()
        text = md_path.read_text(encoding="utf-8")
        for line_no, _is_image, _label, target in extract_markdown_links(text):
            if target.startswith(("http://", "https://", "mailto:", "tel:")):
                continue
            checked_links += 1

            if target.startswith("#"):
                checked_anchors += 1
                anchor = urllib.parse.unquote(target[1:])
                valid_slugs = headings_by_file[md_path.resolve()]
                if anchor not in valid_slugs:
                    issues.append(
                        ValidationIssue(
                            "anchor",
                            rel_md,
                            line_no,
                            f"页内锚点 #{anchor} 未匹配当前文档任何标题",
                        )
                    )
                continue

            file_part, _, fragment = target.partition("#")
            decoded_file = urllib.parse.unquote(file_part)
            resolved_target = (md_path.parent / decoded_file).resolve()
            if not resolved_target.exists():
                issues.append(
                    ValidationIssue(
                        "link",
                        rel_md,
                        line_no,
                        f"本地相对路径不存在: {target}",
                    )
                )
                continue

            referenced_files.add(resolved_target)

            if fragment:
                checked_anchors += 1
                anchor = urllib.parse.unquote(fragment)
                if resolved_target.suffix.lower() == ".md":
                    target_slugs = headings_by_file.get(resolved_target)
                    if target_slugs is None:
                        target_text = resolved_target.read_text(encoding="utf-8")
                        target_slugs = {slug for _, _, _, slug in extract_headings(target_text)}
                        headings_by_file[resolved_target] = target_slugs
                    if anchor not in target_slugs:
                        issues.append(
                            ValidationIssue(
                                "anchor",
                                rel_md,
                                line_no,
                                f"跨文件锚点 {target} 在目标文档中不存在",
                            )
                        )

    # 检查 assets/ 下是否存在未被任何文档引用的孤立图片资源
    assets_dir = repo_root / "assets"
    if assets_dir.is_dir():
        for asset in sorted(assets_dir.iterdir()):
            if (
                asset.is_file()
                and asset.suffix.lower() in ASSET_EXTENSIONS
                and asset.resolve() not in referenced_files
            ):
                issues.append(
                    ValidationIssue(
                        "orphan-asset",
                        asset.relative_to(repo_root).as_posix(),
                        None,
                        "资源文件未被任何 Markdown 文档引用（孤立资源）",
                    )
                )

    # 检查 notes/ 下所有专题笔记是否均被 README.md 引用
    notes_dir = repo_root / "notes"
    readme_path = repo_root / "README.md"
    if notes_dir.is_dir() and readme_path.is_file():
        readme_links = {
            (readme_path.parent / urllib.parse.unquote(t.partition("#")[0])).resolve()
            for _, _, _, t in extract_markdown_links(readme_path.read_text(encoding="utf-8"))
            if not t.startswith(("http://", "https://", "mailto:", "#"))
        }
        for note in sorted(notes_dir.glob("*.md")):
            if note.resolve() not in readme_links:
                issues.append(
                    ValidationIssue(
                        "orphan-note",
                        note.relative_to(repo_root).as_posix(),
                        None,
                        "专题笔记未被 README.md 索引引用",
                    )
                )

    return issues, checked_links, checked_anchors


def _estimate_text_width(content: str, font_size: float) -> float:
    """按中英文字符比例估算 SVG 单行文本渲染像素宽度。"""
    width = 0.0
    for ch in content:
        if ord(ch) > 127:
            width += font_size * 1.0
        elif ch in "ilI1.,:;!| ":
            width += font_size * 0.35
        elif ch in "MW@#%&":
            width += font_size * 0.85
        else:
            width += font_size * 0.58
    return width


def validate_svgs(repo_root: Path) -> tuple[list[ValidationIssue], int]:
    """校验所有 SVG 的 XML 合法性、xmlns/viewBox 规范性与文本边界安全。"""
    issues: list[ValidationIssue] = []
    assets_dir = repo_root / "assets"
    svg_files = sorted(assets_dir.glob("*.svg")) if assets_dir.is_dir() else []

    for svg_path in svg_files:
        rel_svg = svg_path.relative_to(repo_root).as_posix()
        try:
            raw = svg_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            issues.append(ValidationIssue("svg-encoding", rel_svg, None, f"UTF-8 解码失败: {exc}"))
            continue

        if "\ufffd" in raw:
            issues.append(ValidationIssue("svg-encoding", rel_svg, None, "检测到损坏字符 U+FFFD"))

        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            issues.append(ValidationIssue("svg-xml", rel_svg, None, f"非法 XML: {exc}"))
            continue

        # 校验根元素标签与标准命名空间
        if root.tag != f"{{{SVG_NAMESPACE}}}svg":
            issues.append(
                ValidationIssue(
                    "svg-xmlns",
                    rel_svg,
                    None,
                    f"根元素缺少标准 xmlns='{SVG_NAMESPACE}' 或根标签不是 <svg>（实际: {root.tag}）",
                )
            )

        # 校验 viewBox 属性
        viewbox = root.attrib.get("viewBox", "").strip()
        vb_parts = viewbox.split()
        if len(vb_parts) != 4:
            issues.append(ValidationIssue("svg-viewbox", rel_svg, None, f"缺少合法四元 viewBox: '{viewbox}'"))
            continue
        try:
            vb_x, vb_y, vb_w, vb_h = (float(v) for v in vb_parts)
        except ValueError:
            issues.append(ValidationIssue("svg-viewbox", rel_svg, None, f"viewBox 含非数值项: '{viewbox}'"))
            continue

        if vb_w <= 0 or vb_h <= 0:
            issues.append(ValidationIssue("svg-viewbox", rel_svg, None, f"viewBox 宽高必须为正: '{viewbox}'"))
            continue

        # 收集卡片矩形容器并执行文本越界检查（允许 4px 字体渲染容差）
        rects: list[tuple[float, float, float, float]] = []
        for elem in root.iter():
            local_tag = elem.tag.split("}")[-1]
            if local_tag == "rect":
                try:
                    rx = float(elem.attrib.get("x", 0))
                    ry = float(elem.attrib.get("y", 0))
                    rw = float(elem.attrib.get("width", 0))
                    rh = float(elem.attrib.get("height", 0))
                except ValueError:
                    continue
                if 0 < rw < vb_w and rh > 0:
                    rects.append((rx, ry, rx + rw, ry + rh))

        for elem in root.iter():
            local_tag = elem.tag.split("}")[-1]
            if local_tag != "text":
                continue
            txt = "".join(elem.itertext()).strip()
            if not txt:
                continue
            try:
                tx = float(elem.attrib.get("x", 0))
                ty = float(elem.attrib.get("y", 0))
                font_size = float(elem.attrib.get("font-size", 12))
            except ValueError:
                continue

            anchor = elem.attrib.get("text-anchor", "start")
            width = _estimate_text_width(txt, font_size)
            if anchor == "middle":
                x0, x1 = tx - width / 2, tx + width / 2
            elif anchor == "end":
                x0, x1 = tx - width, tx
            else:
                x0, x1 = tx, tx + width

            if x0 < vb_x - 4 or x1 > vb_x + vb_w + 4 or ty < vb_y or ty > vb_y + vb_h + 4:
                issues.append(
                    ValidationIssue(
                        "svg-bounds",
                        rel_svg,
                        None,
                        f"文本超出画布 viewBox 边界 (x=[{x0:.0f},{x1:.0f}], y={ty:.0f}): {txt}",
                    )
                )
                continue

            enclosing = [rc for rc in rects if rc[0] <= tx <= rc[2] and rc[1] <= ty <= rc[3]]
            if enclosing:
                smallest = min(enclosing, key=lambda rc: (rc[2] - rc[0]) * (rc[3] - rc[1]))
                if x0 < smallest[0] - 4 or x1 > smallest[2] + 4:
                    msg = (
                        f"文本可能溢出容器边界 (x=[{x0:.0f},{x1:.0f}], "
                        f"box=[{smallest[0]:.0f},{smallest[2]:.0f}]): {txt}"
                    )
                    issues.append(ValidationIssue("svg-overflow", rel_svg, None, msg))

    return issues, len(svg_files)


def _extract_svg_title_figure_number(svg_path: Path) -> int | None:
    """从 SVG 主标题文本中提取图号（例如 '图 10：...' -> 10）。"""
    try:
        root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    except (OSError, ET.ParseError):
        return None
    for elem in root.iter():
        if elem.tag.split("}")[-1] == "text":
            txt = "".join(elem.itertext()).strip()
            match = FIG_NUM_RE.match(txt)
            if match:
                return int(match.group(1))
    return None


def validate_figure_numbering_and_toc(repo_root: Path) -> list[ValidationIssue]:
    """校验主书图号严格递增、SVG 内部图号与 Markdown 一致、目录与章节标题对齐及笔记返回导航。"""
    issues: list[ValidationIssue] = []
    readme_path = repo_root / "README.md"

    if readme_path.is_file():
        readme_text = readme_path.read_text(encoding="utf-8")
        headings = extract_headings(readme_text)
        heading_by_slug = {slug: strip_markdown_inline(title) for _, _, title, slug in headings}

        # 1. 校验 README 中全部插图的图号连续递增且与 SVG 标题一致
        readme_fig_numbers: list[tuple[int, int]] = []
        for line_no, is_image, alt, target in extract_markdown_links(readme_text):
            if not is_image:
                continue
            alt_match = FIG_NUM_RE.search(alt)
            if alt_match:
                fig_num = int(alt_match.group(1))
                readme_fig_numbers.append((line_no, fig_num))
                target_path = (readme_path.parent / target).resolve()
                if target_path.suffix.lower() == ".svg" and target_path.is_file():
                    svg_num = _extract_svg_title_figure_number(target_path)
                    if svg_num is not None and svg_num != fig_num:
                        issues.append(
                            ValidationIssue(
                                "figure-mismatch",
                                "README.md",
                                line_no,
                                f"Markdown 图注为 '图 {fig_num}'，但 {target} 内部标题为 '图 {svg_num}'",
                            )
                        )

        expected = 1
        for line_no, fig_num in readme_fig_numbers:
            if fig_num != expected:
                issues.append(
                    ValidationIssue(
                        "figure-order",
                        "README.md",
                        line_no,
                        f"主书图号顺序错乱：期望 '图 {expected}'，实际为 '图 {fig_num}'",
                    )
                )
            expected = fig_num + 1

        # 2. 校验目录锚点链接文本与目标标题一致（限 ## 全书章节导航 到下一 ## 标题之间）
        in_toc = False
        for line_no, line in iter_non_code_lines(readme_text):
            h_match = HEADING_RE.match(line)
            if h_match:
                level = len(h_match.group(1))
                title = h_match.group(2).strip()
                if level == 2 and "章节导航" in title:
                    in_toc = True
                    continue
                if level == 2 and in_toc:
                    in_toc = False
            if in_toc:
                for match in LINK_RE.finditer(line):
                    if match.group(1) == "!":
                        continue
                    label = strip_markdown_inline(match.group(2))
                    target = parse_link_target(match.group(3))
                    if target.startswith("#"):
                        slug = urllib.parse.unquote(target[1:])
                        target_title = heading_by_slug.get(slug)
                        if target_title is not None and label != target_title:
                            issues.append(
                                ValidationIssue(
                                    "toc-mismatch",
                                    "README.md",
                                    line_no,
                                    f"目录文本 '{label}' 与目标标题 '{target_title}' 不一致",
                                )
                            )

    # 3. 校验所有专题笔记：SVG 图号一致性 + 必须包含返回 README.md 的导航链接
    notes_dir = repo_root / "notes"
    if notes_dir.is_dir():
        for note_path in sorted(notes_dir.glob("*.md")):
            rel_note = note_path.relative_to(repo_root).as_posix()
            note_text = note_path.read_text(encoding="utf-8")
            links = extract_markdown_links(note_text)
            has_back_to_readme = False
            for line_no, is_image, alt, target in links:
                file_part = target.partition("#")[0]
                if not is_image and file_part.endswith("README.md"):
                    has_back_to_readme = True
                if is_image:
                    alt_match = FIG_NUM_RE.search(alt)
                    target_path = (note_path.parent / file_part).resolve()
                    if alt_match and target_path.suffix.lower() == ".svg" and target_path.is_file():
                        fig_num = int(alt_match.group(1))
                        svg_num = _extract_svg_title_figure_number(target_path)
                        if svg_num is not None and svg_num != fig_num:
                            issues.append(
                                ValidationIssue(
                                    "figure-mismatch",
                                    rel_note,
                                    line_no,
                                    f"Markdown 图注为 '图 {fig_num}'，但 {target} 内部标题为 '图 {svg_num}'",
                                )
                            )
            if not has_back_to_readme:
                issues.append(
                    ValidationIssue(
                        "navigation",
                        rel_note,
                        None,
                        "专题笔记缺少返回 README.md 总目录的导航链接",
                    )
                )

    return issues


def _count_table_columns(line: str) -> int:
    """计算 Markdown 表格单行的列数（忽略转义管道符）。"""
    stripped = line.strip().removeprefix("|").removesuffix("|")
    parts = re.split(r"(?<!\\)\|", stripped)
    return len(parts)


def validate_markdown_and_latex(repo_root: Path) -> list[ValidationIssue]:
    """校验 Markdown 代码块语言标识、表格列对齐与 LaTeX 公式闭合性。"""
    issues: list[ValidationIssue] = []

    for md_path in discover_markdown_files(repo_root):
        rel_md = md_path.relative_to(repo_root).as_posix()
        text = md_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # 1. 围栏代码块语言标识与闭合校验
        in_fence = False
        fence_char = ""
        fence_len = 0
        fence_start_line = 0
        for line_no, line in enumerate(lines, start=1):
            match = FENCE_RE.match(line)
            if not match:
                continue
            marker = match.group(1)
            info = match.group(2).strip()
            if not in_fence:
                in_fence = True
                fence_char = marker[0]
                fence_len = len(marker)
                fence_start_line = line_no
                if not info:
                    issues.append(
                        ValidationIssue(
                            "code-fence",
                            rel_md,
                            line_no,
                            "围栏代码块缺少语言标识（如 python / json / bash）",
                        )
                    )
            elif marker[0] == fence_char and len(marker) >= fence_len and not info:
                in_fence = False
        if in_fence:
            issues.append(
                ValidationIssue("code-fence", rel_md, fence_start_line, "围栏代码块未闭合")
            )

        # 2. LaTeX 块级 $$ 闭合、行内 $ 闭合、标题裸公式检查
        in_display_math = False
        display_start_line = 0
        non_code = list(iter_non_code_lines(text))

        for line_no, line in non_code:
            h_match = HEADING_RE.match(line)
            if h_match and re.search(r"(?<!\\)\$[^$]+\$", h_match.group(2)):
                issues.append(
                    ValidationIssue(
                        "heading-latex",
                        rel_md,
                        line_no,
                        "标题中包含裸 LaTeX 公式（易导致 GitHub 大纲与锚点解析异常）",
                    )
                )

            # 剥离行内反引号代码后再统计 $
            masked = INLINE_CODE_RE.sub("", line)
            dd_tokens = re.findall(r"(?<!\\)\$\$", masked)
            if dd_tokens:
                if len(dd_tokens) % 2 == 1:
                    if not in_display_math:
                        in_display_math = True
                        display_start_line = line_no
                    else:
                        in_display_math = False
                # 移除配对或起止 $$ 后再检查行内 $
                masked = re.sub(r"(?<!\\)\$\$", "", masked)

            if not in_display_math:
                single_dollars = re.findall(r"(?<!\\)\$", masked)
                if len(single_dollars) % 2 != 0:
                    issues.append(
                        ValidationIssue(
                            "latex-inline",
                            rel_md,
                            line_no,
                            f"行内 LaTeX '$' 数量为奇数 ({len(single_dollars)})，疑似未闭合或未转义",
                        )
                    )

        if in_display_math:
            issues.append(
                ValidationIssue(
                    "latex-display",
                    rel_md,
                    display_start_line,
                    "块级 LaTeX 公式 '$$' 未成对闭合",
                )
            )

        # 3. Markdown 表格列数一致性检查
        non_code_map = dict(non_code)
        idx = 0
        while idx < len(non_code):
            line_no, line = non_code[idx]
            if TABLE_SEP_RE.match(line) and idx > 0 and (line_no - 1) in non_code_map:
                header_line = non_code_map[line_no - 1]
                expected_cols = _count_table_columns(header_line)
                sep_cols = _count_table_columns(line)
                if sep_cols != expected_cols:
                    issues.append(
                        ValidationIssue(
                            "table",
                            rel_md,
                            line_no,
                            f"表格分隔行列数 ({sep_cols}) 与表头列数 ({expected_cols}) 不一致",
                        )
                    )
                # 检查后续连续数据行
                row_idx = idx + 1
                prev_line_no = line_no
                while row_idx < len(non_code):
                    cur_line_no, cur_line = non_code[row_idx]
                    if cur_line_no != prev_line_no + 1 or "|" not in cur_line or not cur_line.strip():
                        break
                    row_cols = _count_table_columns(cur_line)
                    if row_cols != expected_cols:
                        issues.append(
                            ValidationIssue(
                                "table",
                                rel_md,
                                cur_line_no,
                                f"表格数据行列数 ({row_cols}) 与表头列数 ({expected_cols}) 不一致",
                            )
                        )
                    prev_line_no = cur_line_no
                    row_idx += 1
                idx = row_idx
                continue
            idx += 1

    return issues


def validate_sensitive_info(repo_root: Path) -> list[ValidationIssue]:
    """扫描 Markdown 与 SVG 文档，防止未脱敏密钥或私网敏感地址泄露。"""
    issues: list[ValidationIssue] = []
    targets: list[Path] = list(discover_markdown_files(repo_root))
    assets_dir = repo_root / "assets"
    if assets_dir.is_dir():
        targets.extend(sorted(assets_dir.glob("*.svg")))

    for file_path in targets:
        rel_path = file_path.relative_to(repo_root).as_posix()
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            for label, pattern in SENSITIVE_PATTERNS:
                if pattern.search(line):
                    issues.append(
                        ValidationIssue(
                            "sensitive-info",
                            rel_path,
                            line_no,
                            f"命中敏感信息模式（{label}），请确认并脱敏处理",
                        )
                    )
    return issues


def run_all_validations(repo_root: Path) -> ValidationReport:
    """运行全量文档与资源质量检查并汇总报告。"""
    root = repo_root.resolve()
    md_files = discover_markdown_files(root)

    link_issues, checked_links, checked_anchors = validate_links_and_assets(root)
    svg_issues, checked_svgs = validate_svgs(root)
    toc_fig_issues = validate_figure_numbering_and_toc(root)
    md_latex_issues = validate_markdown_and_latex(root)
    sensitive_issues = validate_sensitive_info(root)

    all_issues = [
        *link_issues,
        *svg_issues,
        *toc_fig_issues,
        *md_latex_issues,
        *sensitive_issues,
    ]
    return ValidationReport(
        issues=all_issues,
        checked_md_files=len(md_files),
        checked_svg_files=checked_svgs,
        checked_links=checked_links,
        checked_anchors=checked_anchors,
    )


def _ensure_utf8_stdout() -> None:
    """在 Windows 控制台下确保 stdout 使用 UTF-8 编码输出中文。"""
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(OSError, ValueError):
            sys.stdout.reconfigure(encoding="utf-8")
    elif isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    """命令行入口函数。"""
    _ensure_utf8_stdout()
    parser = argparse.ArgumentParser(description="校验多智能体学习笔记仓库的文档与 SVG 完整性")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="仓库根目录路径（默认自动推断为脚本上层目录）",
    )
    args = parser.parse_args(argv)

    report = run_all_validations(args.root)
    print(
        f"质检完成：Markdown={report.checked_md_files} 篇 | "
        f"SVG={report.checked_svg_files} 张 | "
        f"本地链接={report.checked_links} 处 | "
        f"锚点={report.checked_anchors} 处 | "
        f"发现问题={len(report.issues)} 项"
    )
    if report.issues:
        for issue in report.issues:
            print(f"  ✗ {issue.format()}")
        return 1

    print("  ✓ 全部文档、链接、锚点、SVG、LaTeX 与安全隐私检查 100% 通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
