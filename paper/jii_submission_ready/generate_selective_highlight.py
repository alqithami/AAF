#!/usr/bin/env python3
"""Generate an additions-only reviewer copy from submitted and revised LaTeX.

Only added or materially changed text is colored red. Deleted material is
suppressed, unchanged text remains black, and included figures retain their
native colors. Requires `latexdiff` on PATH.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def sanitize_revised(text: str) -> str:
    """Remove the legacy global-highlight wrapper from the clean source."""
    text = re.sub(
        r"\\newif\\ifhighlighted\s*\\highlighted(?:true|false)\s*"
        r"\\ifhighlighted.*?\\fi\s*",
        "",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"\\ifhighlighted\s*\\begin\{center\}.*?\\end\{center\}\s*\\fi\s*",
        "",
        text,
        count=1,
        flags=re.S,
    )
    text = text.replace(r"\begin{revision}", "", 1)
    text = text.replace(r"\end{revision}", "", 1)
    return text


def remove_deleted_blocks(body: str) -> str:
    """Remove balanced latexdiff deletion regions, including FL crossovers."""
    token_re = re.compile(r"\\DIFdel(?:begin|end)(?:FL)?(?![A-Za-z])")
    out: list[str] = []
    last = 0
    depth = 0
    for match in token_re.finditer(body):
        token = match.group(0)
        if "begin" in token:
            if depth == 0:
                out.append(body[last : match.start()])
            depth += 1
        else:
            if depth == 0:
                raise RuntimeError(f"Unmatched deletion end at {match.start()}")
            depth -= 1
            if depth == 0:
                last = match.end()
    if depth:
        raise RuntimeError(f"Unclosed deletion block: depth={depth}")
    out.append(body[last:])
    return "".join(out)


def replace_diff_preamble(text: str) -> str:
    definitions = {
        "DIFadd": r"\providecommand{\DIFadd}[1]{{\protect\color{red}#1}} %DIF PREAMBLE",
        "DIFdel": r"\providecommand{\DIFdel}[1]{} %DIF PREAMBLE",
        "DIFaddbegin": r"\providecommand{\DIFaddbegin}{} %DIF PREAMBLE",
        "DIFaddend": r"\providecommand{\DIFaddend}{} %DIF PREAMBLE",
        "DIFdelbegin": r"\providecommand{\DIFdelbegin}{} %DIF PREAMBLE",
        "DIFdelend": r"\providecommand{\DIFdelend}{} %DIF PREAMBLE",
        "DIFaddFL": r"\providecommand{\DIFaddFL}[1]{{\protect\color{red}#1}} %DIF PREAMBLE",
        "DIFdelFL": r"\providecommand{\DIFdelFL}[1]{} %DIF PREAMBLE",
        "DIFaddbeginFL": r"\providecommand{\DIFaddbeginFL}{} %DIF PREAMBLE",
        "DIFaddendFL": r"\providecommand{\DIFaddendFL}{} %DIF PREAMBLE",
        "DIFdelbeginFL": r"\providecommand{\DIFdelbeginFL}{} %DIF PREAMBLE",
        "DIFdelendFL": r"\providecommand{\DIFdelendFL}{} %DIF PREAMBLE",
    }
    for name, replacement in definitions.items():
        argc = r"\[1\]" if name in {"DIFadd", "DIFdel", "DIFaddFL", "DIFdelFL"} else ""
        pattern = rf"\\providecommand\{{\\{name}\}}{argc}\{{.*?\}}\s*%DIF PREAMBLE"
        text, count = re.subn(pattern, lambda _: replacement, text, count=1, flags=re.S)
        if count != 1:
            raise RuntimeError(f"Could not replace latexdiff macro {name}")
    return text


def generate(original: Path, revised: Path, output: Path) -> None:
    original = original.resolve()
    revised = revised.resolve()
    output = output.resolve()
    if not shutil.which("latexdiff"):
        raise RuntimeError("latexdiff is not available on PATH")

    with tempfile.TemporaryDirectory(prefix="aaf_selective_diff_") as tmp:
        root = Path(tmp)
        old_dir, new_dir = root / "old", root / "new"
        old_dir.mkdir()
        new_dir.mkdir()
        for source_dir, target_dir in [(original.parent, old_dir), (revised.parent, new_dir)]:
            for item in source_dir.glob("*.tex"):
                shutil.copy2(item, target_dir / item.name)

        old_main = old_dir / "main.tex"
        new_main = new_dir / "main.tex"
        old_main.write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
        new_main.write_text(sanitize_revised(revised.read_text(encoding="utf-8")), encoding="utf-8")

        preamble = root / "additions_only_preamble.tex"
        preamble.write_text(
            r"""%DIF ADDITIONS-ONLY RED PREAMBLE
\RequirePackage{xcolor}
\providecommand{\DIFadd}[1]{{\protect\color{red}#1}}
\providecommand{\DIFdel}[1]{}
\providecommand{\DIFaddbegin}{}
\providecommand{\DIFaddend}{}
\providecommand{\DIFdelbegin}{}
\providecommand{\DIFdelend}{}
\providecommand{\DIFaddFL}[1]{{\protect\color{red}#1}}
\providecommand{\DIFdelFL}[1]{}
\providecommand{\DIFaddbeginFL}{}
\providecommand{\DIFaddendFL}{}
\providecommand{\DIFdelbeginFL}{}
\providecommand{\DIFdelendFL}{}
""",
            encoding="utf-8",
        )
        command = [
            "latexdiff", "--flatten", f"--preamble={preamble}",
            "--math-markup=whole", "--graphics-markup=none",
            "--disable-citation-markup", "--ignore-warnings",
            str(old_main), str(new_main),
        ]
        raw = subprocess.run(command, check=True, text=True, capture_output=True).stdout

    raw = replace_diff_preamble(raw)
    marker = r"\begin{document}"
    index = raw.find(marker)
    if index < 0:
        raise RuntimeError("Generated diff has no document body")
    split = index + len(marker)
    head, body = raw[:split], raw[split:]
    body = remove_deleted_blocks(body)
    body = re.sub(r"(?m)^%DIF(?:DELCMD)?(?: <)?.*\n?", "", body)
    body = re.sub(r"(?m)^%DIFAUXCMD.*\n?", "", body)
    note = (
        r"\begin{center}" "\n"
        r"\small\textbf{Revision markup: only text added or materially changed relative to the submitted manuscript is shown in "
        r"\textcolor{red}{red}. Unchanged text and figures remain in their normal colors; deletions are omitted from this reviewer-facing copy.}" "\n"
        r"\end{center}" "\n"
    )
    body = body.replace(r"\maketitle" + "\n", r"\maketitle" + "\n" + note, 1)
    body = body.replace(r"\bibliography{cas-refs}", r"\color{black}\bibliography{cas-refs}", 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(head + body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--revised", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.original, args.revised, args.output)


if __name__ == "__main__":
    main()
