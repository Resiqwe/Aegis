import ast
import os

ignore_dirs = {".venv", "venv", "__pycache__", ".git", ".vscode", "archive"}
nebezpecne_operace = {
    "cpu", "numpy", "tolist", "start_new_operation", 
    "get_battle_state_snapshot", "synchronize", "pos", "teams", "hp", "brains"
}
zamky = {"_engine_lock", "_lock", "lock"}

class SyncAnalyzer(ast.NodeVisitor):
    def __init__(self, filepath):
        self.filepath = filepath
        self.current_class = None
        self.current_func = None
        self.in_lock = False
        self.findings = []

    def visit_ClassDef(self, node):
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node):
        prev_func = self.current_func
        self.current_func = node.name
        self.generic_visit(node)
        self.current_func = prev_func

    def visit_With(self, node):
        is_lock_block = any(
            any(z in ast.dump(item.context_expr) for z in zamky)
            for item in node.items
        )
        prev_lock = self.in_lock
        if is_lock_block:
            self.in_lock = True
        self.generic_visit(node)
        self.in_lock = prev_lock

    def visit_Attribute(self, node):
        if node.attr in nebezpecne_operace:
            lokace = f"{self.current_class}.{self.current_func}" if self.current_class else str(self.current_func)
            stav = "ZAMČENO [OK]" if self.in_lock else "NEZAMČENO [RIZIKO!]"
            self.findings.append(f"{stav} | {self.filepath}:{node.lineno} ({lokace}) -> .{node.attr}")
        self.generic_visit(node)

vsechny_nalezy = []

for root, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    for f in files:
        if f.endswith(".py") and f not in {"scan.py", "rentgen.py", "ast_rentgen.py"}:
            path = os.path.join(root, f)
            with open(path, encoding="utf-8", errors="ignore") as fp:
                try:
                    tree = ast.parse(fp.read(), filename=path)
                    analyzer = SyncAnalyzer(path)
                    analyzer.visit(tree)
                    vsechny_nalezy.extend(analyzer.findings)
                except SyntaxError as e:
                    vsechny_nalezy.append(f"CHYBA SYNTAXE | {path}: {e}")

with open("ast_mapa.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(vsechny_nalezy))

nezamcene = sum(1 for x in vsechny_nalezy if "NEZAMČENO" in x)
print(f"AST hotovo! Celkem {len(vsechny_nalezy)} přístupů, z toho {nezamcene} NEZAMČENÝCH. Výstup je v ast_mapa.txt")
