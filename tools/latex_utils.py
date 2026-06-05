from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple


def _augment_tex_path() -> None:
    raw_path = os.environ.get("PATH", "")
    existing_parts = [part for part in raw_path.split(os.pathsep) if part]
    existing_set = set(existing_parts)

    candidate_dirs = []
    # Common TeX install path on macOS (MacTeX).
    candidate_dirs.append(Path("/Library/TeX/texbin"))
    # Common custom/system binary locations.
    candidate_dirs.extend([Path("/opt/homebrew/bin"), Path("/usr/local/bin"), Path("/usr/texbin")])

    # TeXLive installs can place binaries under versioned folders.
    for root in [Path("/usr/local/texlive"), Path("/opt/texlive")]:
        if not root.exists():
            continue
        for bin_dir in sorted(root.glob("*/bin/*")):
            if bin_dir.is_dir():
                candidate_dirs.append(bin_dir)

    appended_parts = []
    for path_obj in candidate_dirs:
        if not path_obj.exists() or not path_obj.is_dir():
            continue
        path_text = str(path_obj)
        if path_text in existing_set:
            continue
        appended_parts.append(path_text)
        existing_set.add(path_text)

    if appended_parts:
        os.environ["PATH"] = os.pathsep.join(existing_parts + appended_parts)


def detect_latex_compiler() -> Optional[str]:
    _augment_tex_path()
    if shutil.which("latexmk"):
        return "latexmk"
    if shutil.which("pdflatex"):
        return "pdflatex"
    return None


def build_pdf(main_tex: str) -> Tuple[bool, str, Optional[bytes]]:
    compiler = detect_latex_compiler()
    if not compiler:
        return False, "compiler not available", None

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "main.tex"
        tex_path.write_text(main_tex, encoding="utf-8")

        if compiler == "latexmk":
            cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"]
        else:
            cmd = ["pdflatex", "-interaction=nonstopmode", "main.tex"]

        proc = subprocess.run(
            cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            check=False,
        )
        log = proc.stdout + "\n" + proc.stderr
        pdf_path = Path(tmpdir) / "main.pdf"
        if proc.returncode == 0 and pdf_path.exists():
            return True, log, pdf_path.read_bytes()
        return False, log, None
