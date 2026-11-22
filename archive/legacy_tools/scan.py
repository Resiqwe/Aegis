import os, re

patterns = re.compile(r"(threading\.|Lock\(|Queue\(|start_new_operation|get_battle_state_snapshot|\.cpu\(\)|cuda\.synchronize|pygame\.event|tkinter|PyQt|dpg\.)")
ignore_dirs = {".venv", "venv", "__pycache__", ".git", ".vscode"}
nalezy = []

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    for f in files:
        if f.endswith(".py") and f != "rentgen.py":
            path = os.path.join(root, f)
            with open(path, encoding="utf-8", errors="ignore") as fp:
                for i, line in enumerate(fp, 1):
                    if patterns.search(line):
                        nalezy.append(f"{path}:{i}: {line.strip()}")

with open("mapa.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(nalezy))

print(f"Hotovo! Nalezeno {len(nalezy)} uzlů. Vše je uložené v mapa.txt")
