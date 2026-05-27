"""Story 0.3.7 — Notebook logic lint (Risk 10 mitigation).

Pre-commit hook: scans each .ipynb passed on the command line and blocks
commit if any code cell contains a top-level `def ` or `class ` definition.

Why this hook exists:
    All Python logic in this project lives in `src/retention/` so mypy + pytest
    can verify it. Notebook-defined logic bypasses both — "I'll just test it in
    the notebook" becomes a structural integrity hole. This hook makes that
    pattern impossible to commit.

What's allowed in notebooks:
    imports, function calls, assignments, markdown narration, plot generation,
    decorators (lines starting with `@`), and lambda expressions (those are
    callable but not the same anti-pattern — they belong in a single cell as
    a one-shot transformation).

What's blocked:
    Lines matching `^\s*def ` or `^\s*class ` in a code cell's source.

Exit codes:
    0 — all checked notebooks clean
    1 — at least one notebook has a forbidden def/class
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

FORBIDDEN = re.compile(r"^(?:def\s+\w+|class\s+\w+)", re.MULTILINE)


def scan_notebook(path: Path) -> list[tuple[int, str]]:
    """Return a list of (cell_index, offending_line) tuples for a single notebook.

    Empty list = clean. Non-empty = block the commit.
    """
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{path}: invalid notebook JSON — {exc}", file=sys.stderr)
        return [(-1, f"<JSON parse error: {exc}>")]

    findings: list[tuple[int, str]] = []
    for cell_index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        # Notebook source is sometimes a list of lines, sometimes a string.
        if isinstance(source, list):
            source = "".join(source)
        for match in FORBIDDEN.finditer(source):
            line = source[match.start() : source.find("\n", match.start())].strip()
            findings.append((cell_index, line or match.group(0)))
    return findings


def main(argv: list[str]) -> int:
    """Process all .ipynb paths passed as arguments. Return non-zero if any fail."""
    if len(argv) <= 1:
        # Pre-commit invokes with no args when no .ipynb files staged — treat as no-op.
        return 0

    exit_code = 0
    for arg in argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"{path}: not found (skipping)", file=sys.stderr)
            continue
        if path.suffix != ".ipynb":
            continue
        findings = scan_notebook(path)
        if findings:
            exit_code = 1
            print(
                f"\n❌ {path} — found {len(findings)} forbidden def/class definition(s):",
                file=sys.stderr,
            )
            for cell_idx, line in findings:
                print(
                    f"   cell {cell_idx}: {line}",
                    file=sys.stderr,
                )
            print(
                "   → Move this logic to src/retention/ and import it. "
                "Notebook-defined functions/classes bypass mypy + pytest.",
                file=sys.stderr,
            )
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
