"""Compare the packages the code imports with requirements.txt.

    python scripts/check_requirements.py            # from the repo root

Reports
  MISSING  imported by the code but absent from requirements.txt
           (a fresh install would fail with ModuleNotFoundError).
  UNUSED   listed in requirements.txt but never imported directly. These are
           often legitimate (transitive or runtime-only dependencies), so
           review them by hand before deleting anything.

Standard-library modules and the repo's own packages are ignored. Exit code
is 1 when anything is MISSING, so it can gate a commit or CI step.
"""

import ast
import re
import sys
from pathlib import Path

# import name -> PyPI distribution name, where they differ
IMPORT_TO_DIST = {
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "sentence_transformers": "sentence-transformers",
    "rouge_score": "rouge-score",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
}
SKIP_DIRS = {"venv", ".venv", "env", ".git", "__pycache__", "node_modules", ".pytest_cache", "site-packages"}


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def python_files(root: Path):
    for path in root.rglob("*.py"):
        if not SKIP_DIRS & set(path.relative_to(root).parts):
            yield path


def local_names(root: Path) -> set[str]:
    """Names that resolve to the repo's own code: top-level folders plus every
    module file (scripts sometimes import a sibling module by its bare name)."""
    names = {p.stem for p in python_files(root)}
    names |= {p.name for p in root.iterdir() if p.is_dir() and p.name not in SKIP_DIRS}
    return names


def imported_modules(root: Path) -> set[str]:
    found = set()
    for path in python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            print(f"skipping {path}: {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def requirement_names(req_file: Path) -> set[str]:
    names = set()
    for line in req_file.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(normalize(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0]))
    return names


def main(root: str = ".") -> int:
    root = Path(root).resolve()
    req_file = root / "requirements.txt"
    if not req_file.exists():
        print("requirements.txt not found")
        return 1

    third_party = {m for m in imported_modules(root)
                   if m not in sys.stdlib_module_names and m not in local_names(root)}
    needed = {normalize(IMPORT_TO_DIST.get(m, m)): m for m in third_party}
    listed = requirement_names(req_file)

    missing = sorted(d for d in needed if d not in listed)
    unused = sorted(listed - set(needed))

    print(f"imports found: {len(third_party)}   requirements listed: {len(listed)}")
    print("\nMISSING from requirements.txt:" + ("" if missing else " none"))
    for d in missing:
        print(f"  {d}   (imported as '{needed[d]}')")
    print("\nUNUSED (review by hand; may be transitive/runtime):" + ("" if unused else " none"))
    for d in unused:
        print(f"  {d}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))