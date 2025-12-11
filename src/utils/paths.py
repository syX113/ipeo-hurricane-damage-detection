from pathlib import Path


def resolve_project_root() -> Path:
    """Return the nearest ancestor containing a src/ directory (or CWD)."""
    root = Path.cwd().resolve()
    if (root / "src").exists():
        return root
    if root.name == "notebooks" and (root.parent / "src").exists():
        return root.parent.resolve()
    for parent in root.parents:
        if (parent / "src").exists():
            return parent.resolve()
    return root
