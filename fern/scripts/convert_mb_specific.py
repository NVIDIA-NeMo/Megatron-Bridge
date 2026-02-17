#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert Megatron-Bridge-specific MyST/Sphinx syntax: {py:*}, {doc}."""

import argparse
import re
from pathlib import Path

API_DOCS_BASE = "https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs"


def escape_mdx_curly_braces(content: str) -> str:
    """Escape {variable} in code blocks so MDX doesn't parse as JSX."""
    return content.replace("{overrides}", "\\{overrides\\}")


def py_ref_to_api_url(ref: str) -> str:
    """Convert dotted Python ref to API docs URL. e.g. bridge.models -> .../bridge/bridge.models.html"""
    ref = ref.strip().lstrip("~")
    # API structure: apidocs/bridge/<dotted_module>.html
    return f"{API_DOCS_BASE}/bridge/{ref}.html"


def replace_py_role(match: re.Match[str]) -> str:
    """Convert py role to link to API docs (or inline code if not a bridge ref)."""
    text = match.group(1).strip().lstrip("~")
    # Only link bridge.* and megatron.* refs; others stay as inline code
    if text.startswith("bridge.") or text.startswith("megatron."):
        url = py_ref_to_api_url(text)
        return f"[`{text}`]({url})"
    return f"`{text}`"


def convert_py_roles(content: str) -> str:
    """Convert {py:class}, {py:meth}, {py:mod}, {py:attr}, {py:func} to links or inline code."""
    pattern = r"\{py:(?:class|meth|mod|attr|func)\}`([^`<]+?)(?:\s*<[^>]+>)?`"
    return re.sub(pattern, replace_py_role, content)


def convert_doc_roles(content: str) -> str:
    """Convert {doc}`path` to internal links. apidocs paths -> API docs URL."""
    def replace_doc(match: re.Match[str]) -> str:
        path = match.group(1).strip()
        if path.startswith("apidocs/"):
            return f"[API Documentation]({API_DOCS_BASE}/)"
        clean = path.replace("../", "").replace(".md", "").replace(".mdx", "")
        if not clean.startswith("/"):
            clean = "/" + clean
        display = path.split("/")[-1].replace("-", " ").replace("_", " ").title()
        return f"[{display}]({clean})"

    return re.sub(r"\{doc\}`([^`<]+?)(?:\s*<[^>]+>)?`", replace_doc, content)


def convert_file(filepath: Path) -> bool:
    """Convert a single file. Returns True if changes were made."""
    content = filepath.read_text()
    original = content

    content = escape_mdx_curly_braces(content)
    content = convert_py_roles(content)
    content = convert_doc_roles(content)

    if content != original:
        filepath.write_text(content)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Megatron-Bridge-specific syntax (py:class, py:meth)"
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

    changed = []
    for mdx_file in sorted(pages_dir.rglob("*.mdx")):
        if convert_file(mdx_file):
            changed.append(mdx_file.relative_to(pages_dir))
            print(f"  Converted: {mdx_file.relative_to(pages_dir)}")

    print(f"\nConverted {len(changed)} files")


if __name__ == "__main__":
    main()
