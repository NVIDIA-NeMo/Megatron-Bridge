#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert MyST Markdown syntax to Fern MDX components.

Handles: admonitions, dropdowns, tab sets, grid cards, toctree removal,
HTML comments, plus Megatron-Bridge-specific: {image}, {contents},
{literalinclude}, {admonition}, {code-block}, {doctest}.
Also converts <url> and <email> to Markdown links so MDX doesn't parse them as JSX.
Run convert_mb_specific.py first to strip {py:class} and {py:meth} roles.
"""

import argparse
import re
from pathlib import Path


def convert_admonitions(content: str) -> str:
    """Convert MyST admonitions to Fern components."""
    admonition_map = {
        "note": "Note",
        "warning": "Warning",
        "tip": "Tip",
        "important": "Info",
        "seealso": "Note",
        "caution": "Warning",
        "danger": "Warning",
        "attention": "Warning",
        "hint": "Tip",
    }

    for myst_type, fern_component in admonition_map.items():
        pattern = rf"```\{{{myst_type}\}}\s*\n(.*?)```"
        replacement = rf"<{fern_component}>\n\1</{fern_component}>"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)

        pattern = rf":::\{{{myst_type}\}}\s*\n(.*?):::"
        content = re.sub(pattern, replacement, content, flags=re.DOTALL | re.IGNORECASE)

    return content


def convert_admonition_directive(content: str) -> str:
    """Convert {admonition} Title :class: dropdown to Accordion."""
    pattern = r"```\{admonition\}\s+([^\n]+)(?:\s*\n(?::[^\n]+\n)*)?\n(.*?)```"
    def replace(match: re.Match[str]) -> str:
        title = match.group(1).strip().replace('"', "'")
        body = match.group(2).strip()
        return f'<Accordion title="{title}">\n{body}\n</Accordion>'
    return re.sub(pattern, replace, content, flags=re.DOTALL)


def convert_dropdowns(content: str) -> str:
    """Convert MyST dropdowns to Fern Accordion components.

    Handles both fenced ```{dropdown} and directive ::: {dropdown} formats.
    """
    def replace_dropdown(match: re.Match[str]) -> str:
        title = match.group(1).strip()
        body = match.group(2).strip()
        if '"' in title:
            title = title.replace('"', "'")
        return f'<Accordion title="{title}">\n{body}\n</Accordion>'

    # Pattern 1: ```{dropdown} Title\ncontent\n```
    pattern_fenced = r"```\{dropdown\}\s+([^\n]+)\s*\n(.*?)```"
    content = re.sub(pattern_fenced, replace_dropdown, content, flags=re.DOTALL)

    # Pattern 2: ::: {dropdown} Title\ncontent\n:::
    pattern_directive = r":::\s*\{dropdown\}\s+([^\n]+)(?:\s*\n(?::[^\n]+\n)*)?\n(.*?)\n:::\s*\n"
    content = re.sub(pattern_directive, lambda m: replace_dropdown(m) + "\n", content, flags=re.DOTALL)

    return content


def convert_tab_sets(content: str) -> str:
    """Convert MyST tab sets to Fern Tabs components."""
    content = re.sub(r"::::+\s*\{tab-set\}\s*", "<Tabs>\n", content)
    content = re.sub(r"```\{tab-set\}\s*", "<Tabs>\n", content)

    def replace_tab_item(match: re.Match[str]) -> str:
        title = match.group(1).strip()
        return f'<Tab title="{title}">'

    content = re.sub(r"::::*\s*\{tab-item\}\s+([^\n]+)", replace_tab_item, content)
    content = re.sub(r":::*\s*\{tab-item\}\s+([^\n]+)", replace_tab_item, content)

    lines = content.split("\n")
    result = []
    in_tab = False

    for line in lines:
        if '<Tab title="' in line:
            if in_tab:
                result.append("</Tab>\n")
            in_tab = True
            result.append(line)
        elif line.strip() in [":::::", "::::", ":::", "</Tabs>"]:
            if in_tab and line.strip() != "</Tabs>":
                result.append("</Tab>")
                in_tab = False
            if line.strip() in [":::::", "::::"]:
                result.append("</Tabs>")
            else:
                result.append(line)
        else:
            result.append(line)

    content = "\n".join(result)
    content = re.sub(r"\n::::+\n", "\n", content)
    content = re.sub(r"\n:::+\n", "\n", content)
    return content


def convert_grid_cards(content: str) -> str:
    """Convert MyST grid cards to Fern Cards components."""
    content = re.sub(r"::::+\s*\{grid\}[^\n]*\n", "<Cards>\n", content)
    content = re.sub(r"```\{grid\}[^\n]*\n", "<Cards>\n", content)

    def replace_card(match: re.Match[str]) -> str:
        full_match = match.group(0)
        title_match = re.search(r"\{grid-item-card\}\s+(.+?)(?:\n|$)", full_match)
        title = title_match.group(1).strip() if title_match else "Card"
        link_match = re.search(r":link:\s*(\S+)", full_match)
        href = link_match.group(1) if link_match else ""
        if href and href != "apidocs/index":
            if not href.startswith("http"):
                href = "/" + href.replace(".md", "").replace(".mdx", "")
            return f'<Card title="{title}" href="{href}">'
        if href == "apidocs/index":
            return f'<Card title="{title}" href="https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/">'
        return f'<Card title="{title}">'

    content = re.sub(
        r"::::*\s*\{grid-item-card\}[^\n]*(?:\n:link:[^\n]*)?(?:\n:link-type:[^\n]*)?",
        replace_card,
        content,
    )
    content = re.sub(
        r":::*\s*\{grid-item-card\}[^\n]*(?:\n:link:[^\n]*)?(?:\n:link-type:[^\n]*)?",
        replace_card,
        content,
    )

    lines = content.split("\n")
    result = []
    in_card = False

    for line in lines:
        if '<Card title="' in line:
            if in_card:
                result.append("</Card>\n")
            in_card = True
            result.append(line)
        elif line.strip() in [":::::", "::::", ":::", "</Cards>"]:
            if in_card and line.strip() != "</Cards>":
                result.append("\n</Card>")
                in_card = False
            if line.strip() in [":::::", "::::"]:
                result.append("\n</Cards>")
        else:
            result.append(line)

    return "\n".join(result)


def convert_list_table(content: str) -> str:
    """Convert MyST list-table to markdown table.

    Handles ```{list-table} with * - cell format.
    """
    pattern = r"```\{list-table\}[^\n]*(?:\n:[^\n]+)*\n\n(.*?)```"

    def replace_list_table(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        rows: list[list[str]] = []
        for line in body.split("\n"):
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("* -"):
                rows.append([line[3:].strip()])
            elif line.startswith("  -") or line.startswith("-"):
                cell = line.lstrip("- ").strip()
                if rows:
                    rows[-1].append(cell)
                else:
                    rows.append([cell])
        if not rows:
            return match.group(0)
        header = rows[0]
        sep = "| " + " | ".join(["---"] * len(header)) + " |"
        lines_out = ["| " + " | ".join(header) + " |", sep]
        for row in rows[1:]:
            while len(row) < len(header):
                row.append("")
            lines_out.append("| " + " | ".join(row[: len(header)]) + " |")
        return "\n".join(lines_out)

    return re.sub(pattern, replace_list_table, content, flags=re.DOTALL)


def remove_toctree(content: str) -> str:
    """Remove toctree blocks entirely."""
    content = re.sub(r"```\{toctree\}.*?```", "", content, flags=re.DOTALL)
    content = re.sub(r":::\{toctree\}.*?:::", "", content, flags=re.DOTALL)
    return content


def remove_contents(content: str) -> str:
    """Remove {contents} directive (Fern has its own nav)."""
    content = re.sub(r"```\{contents\}.*?```", "", content, flags=re.DOTALL)
    content = re.sub(r":::\{contents\}.*?:::", "", content, flags=re.DOTALL)
    return content


def convert_image(content: str, filepath: Path, repo_root: Path) -> str:
    """Convert {image} path to markdown image. Path relative to current file."""
    pattern = r"```\{image\}\s+([^\s\n]+)(?:\s*\n(?::[^\n]+\n)*)?```"
    def replace(match: re.Match[str]) -> str:
        img_path = match.group(1).strip()
        # Path is relative to current file's dir in docs/; we copy to fern/assets/training/images/
        # output path for update_links: /assets/training/images/filename
        if "images/" in img_path:
            img_name = img_path.split("images/")[-1]
            return f"![{img_name}](/assets/training/images/{img_name})"
        return f"![{img_path}]({img_path})"
    return re.sub(pattern, replace, content)


def convert_literalinclude(content: str, filepath: Path, repo_root: Path) -> str:
    """Convert {literalinclude} to fenced code block. Inlines full file."""
    pattern = r"```\{literalinclude\}\s+([^\s\n]+)(?:\s*\n(?::[^\n]+\n)*)?\s*```"
    def replace(match: re.Match[str]) -> str:
        inc_path = match.group(1).strip()
        # Path is relative to docs/ (e.g. ../src/megatron/...)
        resolved = (repo_root / "docs" / inc_path).resolve()
        if not resolved.exists():
            resolved = (repo_root / inc_path.replace("../", "")).resolve()
        if not resolved.exists():
            return f"<!-- literalinclude not found: {inc_path} -->"
        lang = "python" if resolved.suffix == ".py" else ""
        try:
            body = resolved.read_text()
        except Exception:
            return f"<!-- Error reading {inc_path} -->"
        return f"```{lang}\n{body}\n```"
    return re.sub(pattern, replace, content)


def convert_code_block(content: str) -> str:
    """Convert {code-block} lang to standard ```lang."""
    pattern = r"```\{code-block\}\s+(\w+)(?:\s*\n(?::[^\n]+\n)*)?\n(.*?)```"
    def replace(match: re.Match[str]) -> str:
        lang = match.group(1)
        body = match.group(2).rstrip()
        return f"```{lang}\n{body}\n```"
    return re.sub(pattern, replace, content, flags=re.DOTALL)


def convert_doctest(content: str) -> str:
    """Convert {doctest} to standard code block."""
    pattern = r"```\{doctest\}\s*\n(.*?)```"
    def replace(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        return f"```python\n{body}\n```"
    return re.sub(pattern, replace, content, flags=re.DOTALL)


def escape_sphinx_doc_refs(content: str) -> str:
    """Escape Sphinx doc refs like <project:apidocs/index.rst> that MDX parses as JSX."""
    content = re.sub(
        r"<project:apidocs/index\.rst>",
        "[API Documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/)",
        content,
    )
    return content


def convert_angle_bracket_urls_and_emails(content: str) -> str:
    """Convert <url> and <email> to Markdown links so MDX doesn't parse them as JSX tags."""
    # <https://...> or <http://...> -> [url](url)
    content = re.sub(
        r"<(https?://[^>]+)>",
        r"[\1](\1)",
        content,
    )
    # <email@domain.tld> -> [email](mailto:email)
    content = re.sub(
        r"<([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})>",
        r"[\1](mailto:\1)",
        content,
    )
    return content


def convert_html_comments(content: str) -> str:
    """Convert HTML comments to JSX comments."""
    return re.sub(r"<!--\s*(.*?)\s*-->", r"{/* \1 */}", content, flags=re.DOTALL)


def remove_directive_options(content: str) -> str:
    """Remove MyST directive options."""
    for opt in [
        ":icon:", ":class:", ":columns:", ":gutter:", ":margin:", ":padding:",
        ":link-type:", ":maxdepth:", ":titlesonly:", ":hidden:", ":link:",
        ":caption:", ":language:", ":pyobject:", ":linenos:", ":emphasize-lines:",
        ":width:", ":align:", ":relative-docs:",
    ]:
        content = re.sub(rf"\n{re.escape(opt)}[^\n]*", "", content)
    return content


def fix_malformed_tags(content: str) -> str:
    """Fix common malformed tag issues."""
    content = re.sub(r'title=""', 'title="Details"', content)
    content = re.sub(
        r"<(Note|Warning|Tip|Info)([^>]*)/>\s*\n([^<]+)",
        r"<\1\2>\n\3</\1>",
        content,
    )
    return content


def clean_multiple_newlines(content: str) -> str:
    """Clean up excessive newlines."""
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip() + "\n"


def convert_file(filepath: Path, repo_root: Path) -> bool:
    """Convert a single file. Returns True if changes were made."""
    content = filepath.read_text()
    original = content

    content = convert_admonitions(content)
    content = convert_admonition_directive(content)
    content = convert_dropdowns(content)
    content = convert_grid_cards(content)
    content = convert_tab_sets(content)
    content = convert_list_table(content)
    content = remove_toctree(content)
    content = remove_contents(content)
    content = convert_image(content, filepath, repo_root)
    content = convert_literalinclude(content, filepath, repo_root)
    content = convert_code_block(content)
    content = convert_doctest(content)
    content = escape_sphinx_doc_refs(content)
    content = convert_angle_bracket_urls_and_emails(content)
    content = convert_html_comments(content)
    content = remove_directive_options(content)
    content = fix_malformed_tags(content)
    content = clean_multiple_newlines(content)

    if content != original:
        filepath.write_text(content)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MyST syntax to Fern MDX in pages directory"
    )
    parser.add_argument(
        "pages_dir",
        type=Path,
        help="Path to pages directory (e.g. fern/v0.2.0/pages)",
    )
    args = parser.parse_args()

    pages_dir = args.pages_dir.resolve()
    if not pages_dir.exists():
        raise SystemExit(f"Error: pages directory not found at {pages_dir}")

    repo_root = pages_dir.parent.parent.parent

    changed = []
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        if convert_file(mdx_file, repo_root):
            changed.append(mdx_file.relative_to(pages_dir))
            print(f"  Converted: {mdx_file.relative_to(pages_dir)}")

    print(f"\nConverted {len(changed)} files")


if __name__ == "__main__":
    main()
