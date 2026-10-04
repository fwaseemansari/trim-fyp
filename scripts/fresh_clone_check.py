"""Fresh-clone regression: does the COMMITTED repo install and pass its tests?

    python scripts/fresh_clone_check.py                 # full check
    python scripts/fresh_clone_check.py --skip-install  # reuse current Python

It clones the repo into a temporary folder (so only committed files exist,
exactly what a teammate or examiner would get), creates a clean virtualenv,
installs requirements.txt, and runs pytest. Commit your work first.

Checks, in order:
  1. secrets: .env / key files must not be tracked by git
  2. clone the committed state
  3. create venv + pip install -r requirements.txt   (skipped with --skip-install)
  4. pytest in the clone; tests that need API keys or untracked data will fail
     here, which is the point: they show what a new machine is missing.

Use --pytest-args to narrow the run, e.g.
    --pytest-args "-q --ignore=pipeline/test_pipeline.py"
"""

import argparse
import shlex
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

SECRET_PATTERNS = (".env", ".pem", ".key")


def run(cmd, cwd=None, check=True):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=check)


def tracked_secrets(repo: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines()
            if Path(f).name == ".env" or Path(f).name.startswith(".env.") and not f.endswith(".example")
            or f.endswith((".pem", ".key"))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=".", help="path to the git repo (default: current folder)")
    ap.add_argument("--skip-install", action="store_true",
                    help="use the current Python instead of a fresh venv + pip install")
    ap.add_argument("--pytest-args", default="-q", help="arguments passed to pytest")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()

    print("== 1. secrets check ==")
    secrets = tracked_secrets(repo)
    if secrets:
        print("FAIL: secret-like files are tracked by git:", secrets)
        print("Remove them from the index (git rm --cached <file>), add to .gitignore, "
              "and rotate any exposed keys.")
        return 1
    print("ok: no .env / key files tracked")

    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
    if dirty.strip():
        print("note: uncommitted changes exist; the clone will NOT contain them:")
        print(dirty)

    with tempfile.TemporaryDirectory(prefix="trim_fresh_") as tmp:
        clone = Path(tmp) / "clone"
        print("== 2. clone ==")
        run(["git", "clone", "--quiet", str(repo), str(clone)])

        python = sys.executable
        if not args.skip_install:
            print("== 3. venv + install ==")
            venv_dir = Path(tmp) / "venv"
            venv.create(venv_dir, with_pip=True)
            python = str(venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
            run([python, "-m", "pip", "install", "--quiet", "-r", "requirements.txt"], cwd=clone)
        else:
            print("== 3. install skipped ==")

        print("== 4. pytest ==")
        result = run([python, "-m", "pytest", *shlex.split(args.pytest_args)], cwd=clone, check=False)
        print("\nRESULT:", "PASS" if result.returncode == 0 else f"FAIL (pytest exit {result.returncode})")
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())