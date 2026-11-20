"""
scripts/tools/xray.py - Statický AST rentgen projektu Cyber-Tactics 2.0.

Poskytuje:
  - Top-N nejsložitějších funkcí s vnitřními sekčními kotvami (pro >100 LOC)
  - Detekci GPU anti-patternů (HOT_LOOP_SYNC uvnitř for/while smyček v hot-path)
  - Mapu READS / WRITES atributů self.* pro metody zadané třídy (--deps)
  - Kontrolu F401 (ruff nebo vlastní AST fallback), tichých except, zámků a mrtvých metod
"""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

EXCLUDED_PARTS = {"archive", ".venv", "venv", "__pycache__", ".idea", ".git", "scripts", "telemetry", "docs", "plany"}
GPU_SYNC_ATTRS = {"item", "cpu", "numpy", "tolist", "any", "all"}
HOT_PATH_PREFIXES = ("src/engine_core.py", "src/engine/")


def _iter_active_py_files(root: Path) -> list[tuple[str, Path]]:
    result = []
    for p in sorted(root.rglob("*.py")):
        if any(part in EXCLUDED_PARTS for part in p.parts):
            continue
        result.append((p.relative_to(root).as_posix(), p))
    return result


def _cyclomatic_complexity(node: ast.AST) -> int:
    c = 1
    for n in ast.walk(node):
        if isinstance(n, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.With, ast.AsyncWith, ast.IfExp, ast.Assert)):
            c += 1
        elif isinstance(n, ast.BoolOp):
            c += len(n.values) - 1
        elif isinstance(n, ast.comprehension):
            c += 1 + len(n.ifs)
    return c


def _max_nesting_depth(node: ast.AST, depth: int = 0) -> int:
    best = depth
    for child in ast.iter_child_nodes(node):
        inc = 1 if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)) else 0
        best = max(best, _max_nesting_depth(child, depth + inc))
    return best


def _extract_section_anchors(lines: list[str], start_line: int, end_line: int, max_anchors: int = 8) -> list[str]:
    anchors: list[str] = []
    seen_lines: set[int] = set()
    for ln in range(start_line + 1, min(end_line, len(lines)) + 1):
        s = lines[ln - 1].strip()
        if not s.startswith("#"):
            continue
        # Hledáme sekční nadpisy typu "# 1. ...", "# ====", "# --- ..."
        if re.match(r"^#\s*(?:[=─-]{4,}|[0-9]+\.\s+\S)", s):
            label = re.sub(r"^#\s*[=─-]*\s*", "", s)
            if not label and ln < end_line:
                nxt = lines[ln].strip()
                if nxt.startswith("#"):
                    label = re.sub(r"^#\s*", "", nxt)
                    ln = ln + 1
            label = re.sub(r"[^A-Za-z0-9áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ_ /:-]", "", label).strip()
            if label and ln not in seen_lines:
                seen_lines.add(ln)
                short = label[:28].rstrip()
                anchors.append(f"{ln}:{short}")
                if len(anchors) >= max_anchors:
                    break
    return anchors


def _check_f401_ruff(root: Path, files: list[tuple[str, Path]]) -> list[dict[str, Any]] | None:
    ruff_bin = shutil.which("ruff")
    if not ruff_bin:
        venv_ruff = root / ".venv" / "Scripts" / "ruff.exe"
        if venv_ruff.exists():
            ruff_bin = str(venv_ruff)
    if not ruff_bin:
        return None
    try:
        targets = [str(p) for _, p in files]
        proc = subprocess.run(
            [ruff_bin, "check", "--select", "F401", "--output-format", "json", *targets],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=15,
        )
        data = json.loads(proc.stdout or "[]")
        out = []
        for item in data:
            rel = Path(item["filename"]).resolve().relative_to(root.resolve()).as_posix()
            out.append({
                "file": rel,
                "line": item["location"]["row"],
                "symbol": item.get("message", ""),
            })
        return out
    except Exception:
        return None


def _check_f401_ast(files: list[tuple[str, Path]], parsed: dict[str, ast.AST]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for rel, _ in files:
        if rel.endswith("__init__.py"):
            continue
        tree = parsed.get(rel)
        if tree is None:
            continue
        exported: set[str] = set()
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exported.add(elt.value)

        imported: dict[str, int] = {}
        used: set[str] = set(exported)

        def visit_imports(node: ast.AST, in_type_checking: bool = False):
            if isinstance(node, ast.If):
                test_txt = ast.unparse(node.test) if hasattr(ast, "unparse") else ""
                is_tc = "TYPE_CHECKING" in test_txt
                for ch in node.body:
                    visit_imports(ch, in_type_checking or is_tc)
                for ch in node.orelse:
                    visit_imports(ch, in_type_checking)
                return
            if not in_type_checking:
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imported[a.asname or a.name.split(".")[0]] = node.lineno
                elif isinstance(node, ast.ImportFrom):
                    if node.module != "__future__":
                        for a in node.names:
                            if a.name != "*":
                                imported[a.asname or a.name] = node.lineno
            for ch in ast.iter_child_nodes(node):
                if isinstance(node, ast.If):
                    continue
                visit_imports(ch, in_type_checking)

        visit_imports(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                used.add(n.id)

        for sym, ln in imported.items():
            if sym not in used:
                issues.append({"file": rel, "line": ln, "symbol": sym})
    return issues


def _find_gpu_syncs(rel: str, tree: ast.AST) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def walk_node(node: ast.AST, fn_name: str, in_loop: bool):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_name = node.name
        now_loop = in_loop or isinstance(node, (ast.For, ast.AsyncFor, ast.While))

        if isinstance(node, ast.Call):
            kind = None
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in GPU_SYNC_ATTRS and len(node.args) == 0:
                    kind = f".{node.func.attr}()"
                elif node.func.attr == "synchronize":
                    kind = "torch.cuda.synchronize()"
                elif node.func.attr == "where" and len(node.args) == 1:
                    kind = "torch.where(mask)"
            if kind:
                findings.append({
                    "file": rel,
                    "line": getattr(node, "lineno", 0),
                    "func": fn_name,
                    "kind": kind,
                    "in_loop": now_loop,
                    "hot_loop_sync": now_loop and kind in (".item()", ".cpu()", ".numpy()", ".tolist()", "torch.cuda.synchronize()"),
                })

        for ch in ast.iter_child_nodes(node):
            walk_node(ch, fn_name, now_loop)

    walk_node(tree, "<module>", False)
    return findings


def _find_locks(rel: str, tree: ast.AST) -> list[dict[str, Any]]:
    locks: list[dict[str, Any]] = []

    def walk_node(node: ast.AST, depth: int):
        is_lock_with = False
        lock_name = ""
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                expr = ast.unparse(item.context_expr) if hasattr(ast, "unparse") else ""
                if "lock" in expr.lower():
                    is_lock_with = True
                    lock_name = expr
                    break
        if is_lock_with:
            locks.append({
                "file": rel,
                "line": node.lineno,
                "lock": lock_name,
                "nested": depth > 0,
            })
        for ch in ast.iter_child_nodes(node):
            walk_node(ch, depth + (1 if is_lock_with else 0))

    walk_node(tree, 0)
    return locks


def _extract_class_deps(root: Path, files: list[tuple[str, Path]], target_class: str) -> dict[str, Any]:
    for rel, p in files:
        src = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == target_class:
                methods: list[dict[str, Any]] = []
                for item in node.body:
                    if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    reads: set[str] = set()
                    writes: set[str] = set()

                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self":
                            if isinstance(sub.ctx, ast.Store):
                                writes.add(sub.attr)
                            elif isinstance(sub.ctx, ast.Load):
                                reads.add(sub.attr)
                        elif isinstance(sub, ast.AugAssign):
                            t = sub.target
                            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                                reads.add(t.attr)
                                writes.add(t.attr)
                            elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute):
                                if isinstance(t.value.value, ast.Name) and t.value.value.id == "self":
                                    reads.add(t.value.attr)
                                    writes.add(t.value.attr)
                        elif isinstance(sub, ast.Assign):
                            for t in sub.targets:
                                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Attribute):
                                    if isinstance(t.value.value, ast.Name) and t.value.value.id == "self":
                                        writes.add(t.value.attr)
                        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                            # In-place PyTorch/list mutace self.x.index_add_ / append / clamp_ ...
                            if sub.func.attr.endswith("_") or sub.func.attr in ("append", "pop", "clear", "extend"):
                                recv = sub.func.value
                                if isinstance(recv, ast.Attribute) and isinstance(recv.value, ast.Name) and recv.value.id == "self":
                                    writes.add(recv.attr)

                    methods.append({
                        "name": item.name,
                        "lines": f"{item.lineno}-{item.end_lineno or item.lineno}",
                        "loc": (item.end_lineno or item.lineno) - item.lineno + 1,
                        "reads": sorted(reads),
                        "writes": sorted(writes),
                    })
                return {"class": target_class, "file": rel, "methods": methods}
    return {"class": target_class, "file": None, "methods": []}


def run_xray(
    root: Path,
    compact: bool = False,
    as_json: bool = False,
    deps_class: str | None = None,
    strict: bool = False,
    strict_all: bool = False,
) -> int:
    files = _iter_active_py_files(root)

    if deps_class:
        deps = _extract_class_deps(root, files, deps_class)
        if as_json:
            print(json.dumps(deps, indent=2, ensure_ascii=False))
        else:
            if not deps["file"]:
                print(f"[AEGIS XRAY] Třída '{deps_class}' nenalezena.")
                return 1
            print(f"[AEGIS XRAY DEPS] {deps['class']} ({deps['file']}) | methods={len(deps['methods'])}")
            for m in deps["methods"]:
                r_str = ", ".join(m["reads"][:10]) + (" ..." if len(m["reads"]) > 10 else "")
                w_str = ", ".join(m["writes"][:10]) + (" ..." if len(m["writes"]) > 10 else "")
                print(f"  {m['name']} ({m['lines']}, {m['loc']} LOC)")
                print(f"    READS ({len(m['reads'])}):  {r_str or '-'}")
                print(f"    WRITES ({len(m['writes'])}): {w_str or '-'}")
        return 0

    parsed: dict[str, ast.AST] = {}
    sources: dict[str, str] = {}
    total_loc = 0
    funcs_info: list[dict[str, Any]] = []
    silent_excepts: list[dict[str, Any]] = []
    gpu_syncs: list[dict[str, Any]] = []
    locks: list[dict[str, Any]] = []

    for rel, p in files:
        src = p.read_text(encoding="utf-8", errors="replace")
        sources[rel] = src
        lines = src.splitlines()
        total_loc += len(lines)
        tree = ast.parse(src, filename=rel)
        parsed[rel] = tree

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end_ln = node.end_lineno or node.lineno
                floc = end_ln - node.lineno + 1
                cc = _cyclomatic_complexity(node)
                depth = _max_nesting_depth(node)
                anchors = _extract_section_anchors(lines, node.lineno, end_ln) if floc >= 100 else []
                funcs_info.append({
                    "file": rel,
                    "name": node.name,
                    "start": node.lineno,
                    "end": end_ln,
                    "loc": floc,
                    "cc": cc,
                    "depth": depth,
                    "anchors": anchors,
                })
            elif isinstance(node, ast.ExceptHandler):
                is_bare = node.type is None
                is_broad = isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException")
                is_silent = len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Continue))
                if is_bare or (is_broad and is_silent):
                    silent_excepts.append({"file": rel, "line": node.lineno})

        if rel.startswith(HOT_PATH_PREFIXES):
            gpu_syncs.extend(_find_gpu_syncs(rel, tree))
        locks.extend(_find_locks(rel, tree))

    # Detekce nevolaných veřejných metod na subsystémech v src/engine/*
    all_text = "\n".join(sources.values())
    dead_methods: list[dict[str, Any]] = []
    for rel, tree in parsed.items():
        if not rel.startswith("src/engine/") or rel.endswith("__init__.py"):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                        # Počet výskytů `.method_name(` mimo vlastní definici
                        refs = len(re.findall(rf"\.{re.escape(item.name)}\s*\(", all_text))
                        if refs == 0:
                            dead_methods.append({
                                "file": rel,
                                "line": item.lineno,
                                "method": f"{node.name}.{item.name}",
                            })

    f401 = _check_f401_ruff(root, files)
    if f401 is None:
        f401 = _check_f401_ast(files, parsed)

    hot_loop_syncs = [g for g in gpu_syncs if g["hot_loop_sync"]]
    core_hot_loop_syncs = [
        g for g in hot_loop_syncs
        if g["file"] == "src/engine_core.py" and g["func"] in ("_step_internal", "_atomic_commit_pipeline")
    ]
    top_funcs = sorted(funcs_info, key=lambda x: (x["loc"], x["cc"]), reverse=True)[:8]

    report = {
        "files": len(files),
        "loc": total_loc,
        "f401": f401,
        "hot_loop_syncs": hot_loop_syncs,
        "core_hot_loop_syncs": core_hot_loop_syncs,
        "silent_excepts": silent_excepts,
        "dead_methods": dead_methods,
        "top_complexity": top_funcs,
        "locks": locks,
    }

    if as_json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(
            f"[AEGIS XRAY] files={len(files)} loc={total_loc} f401={len(f401)} "
            f"core_hot_loop_sync={len(core_hot_loop_syncs)} all_hot_loop_sync={len(hot_loop_syncs)} "
            f"silent_except={len(silent_excepts)}"
        )
        print("TOP_COMPLEXITY (lines / LOC / CC / depth):")
        limit = 5 if compact else 8
        for idx, fn in enumerate(top_funcs[:limit], 1):
            print(f"  {idx}. {fn['file']}:{fn['start']}-{fn['end']} {fn['name']} (loc={fn['loc']}, cc={fn['cc']}, depth={fn['depth']})")
            if fn["anchors"]:
                print(f"     anchors: {' | '.join(fn['anchors'][:6])}")

        if hot_loop_syncs:
            print("HOT_LOOP_SYNCS (.item/.cpu/.numpy/sync inside for/while):")
            for g in hot_loop_syncs[: (8 if compact else 20)]:
                print(f"  {g['file']}:{g['line']} {g['func']}() -> {g['kind']}")

        if dead_methods:
            print("DEAD_METHODS (0 external refs):")
            for dm in dead_methods[:8]:
                print(f"  {dm['file']}:{dm['line']} {dm['method']}")

        if f401:
            print("UNUSED_IMPORTS (F401):")
            for u in f401[:10]:
                print(f"  {u['file']}:{u['line']} {u['symbol']}")

        by_file_locks: dict[str, list[str]] = defaultdict(list)
        for lk in locks:
            tag = f"{lk['line']}({lk['lock']}{',NESTED' if lk['nested'] else ''})"
            by_file_locks[lk["file"]].append(tag)
        lock_summary = " | ".join(f"{f}:{','.join(v[:3])}" for f, v in sorted(by_file_locks.items())[:6])
        print(f"LOCKS: {lock_summary}")

    if strict_all and (len(f401) > 0 or len(hot_loop_syncs) > 0):
        return 1
    if strict and (len(f401) > 0 or len(core_hot_loop_syncs) > 0):
        return 1
    return 0
