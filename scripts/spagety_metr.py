import ast
from pathlib import Path

IGNORE_DIRS = {".venv", "venv", "__pycache__", ".git", "archive", "telemetry", "logs", "data"}
IGNORE_FILES = {"spagety_metr.py", "ast_rentgen.py", "scan.py"}

def analyze_project(root="."):
    stats = []
    for path in Path(root).rglob("*.py"):
        if any(part.lower() in IGNORE_DIRS for part in path.parts):
            continue
        if path.name in IGNORE_FILES or "telemetry" in path.name.lower():
            continue

        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        total_lines = len(lines)
        sloc = sum(1 for l in lines if l.strip() and not l.strip().startswith("#"))

        try:
            tree = ast.parse(text)
            funcs = sum(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
            classes = sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
        except SyntaxError:
            funcs = classes = 0

        stats.append((sloc, total_lines, funcs, classes, str(path)))

    stats.sort(reverse=True)

    print(f"{'SOUBOR':<42} | {'ČISTÝ KÓD':>9} | {'CELKEM':>7} | {'TŘÍDY':>5} | {'FUNKCE':>6}")
    print("-" * 80)
    for sloc, total, funcs, classes, p in stats:
        print(f"{p:<42} | {sloc:>9} | {total:>7} | {classes:>5} | {funcs:>6}")

    print("-" * 80)
    print(f"{'CELKEM PRODUKČNÍ JÁDRO (' + str(len(stats)) + ' souborů)':<42} | "
          f"{sum(s[0] for s in stats):>9} | {sum(s[1] for s in stats):>7} | "
          f"{sum(s[3] for s in stats):>5} | {sum(s[2] for s in stats):>6}")

if __name__ == "__main__":
    analyze_project()