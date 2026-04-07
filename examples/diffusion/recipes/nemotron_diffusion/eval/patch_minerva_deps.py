#!/usr/bin/env python3
"""Patch lm-eval minerva_math and sympy to accept antlr4 4.9.x (needed for omegaconf compat)."""
import pathlib
import subprocess
import sys


def _patch_file(filepath, old, new, label):
    """Replace old with new in filepath if old is found."""
    path = pathlib.Path(filepath)
    if not path.exists():
        return
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new))
        print(f"Patched {label}: {path}")


def main():
    # Install lm-eval[math] and pin antlr4 back to 4.9.3
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lm-eval[math]==0.4.10", "-q", "--no-cache-dir"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "antlr4-python3-runtime==4.9.3", "-q", "--no-cache-dir"])

    # Relax the antlr4 version check in minerva_math/utils.py
    try:
        import lm_eval.tasks.minerva_math as _m

        _patch_file(
            pathlib.Path(_m.__path__[0]) / "utils.py",
            "startswith(\"4.11\")",
            "startswith((\"4.9\", \"4.11\", \"4.13\"))",
            "minerva_math utils.py",
        )
    except Exception as e:
        print(f"Warning: could not patch minerva_math: {e}")

    # Relax the antlr4 version check in sympy's LaTeX parser
    try:
        import sympy.parsing.latex._parse_latex_antlr as _s

        _patch_file(
            _s.__file__,
            "startswith('4.11')",
            "startswith(('4.9', '4.11', '4.13'))",
            "sympy _parse_latex_antlr",
        )
    except Exception as e:
        print(f"Warning: could not patch sympy: {e}")


if __name__ == "__main__":
    main()
