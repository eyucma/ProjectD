"""
script for finding path to anchor
"""

from pathlib import Path


def find_project_root(
    anchor: str | Path = "data", start_path: str | Path | None = None
) -> Path:
    """
    Finds nearest parent containing anchor from start_path
    """
    if not start_path is None:
        path = Path(start_path).resolve()
        if path.is_file():
            path = path.parent
    else:
        path = Path(__file__).resolve().parent

    # Standard search loop...
    for parent in [path] + list(path.parents):
        if (parent / anchor).exists():
            return parent
    raise RuntimeError(f"Project root not found: missing '{anchor}'")
