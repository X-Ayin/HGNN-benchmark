from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _candidate_sphinx_roots() -> Iterable[Path]:
    repo_root = Path(__file__).resolve().parent.parent.parent
    yield repo_root / "SPHINX-main"
    # Common typo kept as fallback to reduce setup friction.
    yield repo_root / "SPINX-main"
    yield repo_root / "SPINX-mian"


def resolve_sphinx_path() -> Path:
    for candidate in _candidate_sphinx_roots():
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in _candidate_sphinx_roots())
    raise FileNotFoundError(
        f"Cannot locate SPHINX source folder. Tried: {searched}. "
        "Place the repository under workspace root as 'SPHINX-main'."
    )
