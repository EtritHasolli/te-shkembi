import sys
from pathlib import Path

MAX_PER_DIR = 10

def walk(path, prefix=""):
    entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    for i, p in enumerate(entries[:MAX_PER_DIR]):
        is_last = i == min(len(entries), MAX_PER_DIR) - 1
        connector = "`-- " if is_last else "|-- "
        print(prefix + connector + p.name)
        if p.is_dir():
            extension = "    " if is_last else "|   "
            walk(p, prefix + extension)
    if len(entries) > MAX_PER_DIR:
        print(prefix + "`-- ...")

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python tree_folder.py <path>")
        return 1

    root = Path(sys.argv[1])
    print(root.name)
    walk(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
