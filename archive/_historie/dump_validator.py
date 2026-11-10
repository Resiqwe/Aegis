import hashlib
import json
from pathlib import Path

# Vždy vedle skriptu
DUMP_DIR = Path(__file__).resolve().parent / "dumps"

def get_valid_dumps(quiet: bool = True) -> list:
    """
    Potichu projde složku, ověří integritu párů včetně SHA-256 haše vah
    a vrátí seznam cesty k validním meta souborům.
    """
    if not DUMP_DIR.exists():
        return []

    meta_files = sorted(DUMP_DIR.glob("*_meta.json"))
    valid_files = []

    for meta_path in meta_files:
        base_prefix = meta_path.name.replace("_meta.json", "")
        weights_path = DUMP_DIR / f"{base_prefix}_weights.dat"

        # 1. Kontrola existence páru
        if not weights_path.exists():
            if not quiet: print(f"[VALIDÁTOR] Chybí váhy pro {base_prefix}")
            continue
            
        # 2. Kontrola nulové velikosti
        if meta_path.stat().st_size == 0 or weights_path.stat().st_size == 0:
            if not quiet: print(f"[VALIDÁTOR] Prázdný soubor u {base_prefix}")
            continue
            
        # 3. Kontrola validního JSONu
        try:
            with open(meta_path, "r", encoding="utf-8-sig") as f:
                meta = json.load(f)
        except Exception as e:
            if not quiet: print(f"[VALIDÁTOR] Poškozený JSON u {base_prefix} ({e})")
            continue

        # 4. Kontrola OBSAHU vah: každá hodnota musí být číslo
        try:
            with open(weights_path, "r", encoding="utf-8-sig") as f:
                hodnoty = f.read().split()
            if not hodnoty:
                raise ValueError("žádné hodnoty")
            [float(h) for h in hodnoty]
        except Exception as e:
            if not quiet: print(f"[VALIDÁTOR] Váhy u {base_prefix} nejsou čísla ({e})")
            continue

        # 5. Kontrola SHA-256 haše (Ověření, že s váhami nebylo manipulováno)
        try:
            with open(weights_path, "rb") as wf:
                current_hash = hashlib.sha256(wf.read()).hexdigest()
            
            expected_hash = meta.get("weights_hash")
            if expected_hash and current_hash != expected_hash:
                if not quiet: print(f"[VALIDÁTOR] Narušení integrity! Soubor vah {weights_path.name} byl modifikován (haš nesouhlasí)!")
                continue
        except Exception as e:
            if not quiet: print(f"[VALIDÁTOR] Chyba při výpočtu haše u {base_prefix} ({e})")
            continue

        valid_files.append(meta_path)

    return valid_files

def validate_all_dumps_cli():
    """Interaktivní CLI režim pro ruční spuštění v terminálu."""
    if not DUMP_DIR.exists():
        print("[VALIDÁTOR] Chyba: Adresář ./dumps neexistuje!")
        return False

    print(f"[VALIDÁTOR] Zahajuji kontrolu včetně integritních hashů...")
    meta_files = sorted(DUMP_DIR.glob("*_meta.json"))
    valid = get_valid_dumps(quiet=False)
    
    print(f"\n[VALIDÁTOR] Hotovo. Zkontrolováno {len(meta_files)} artefaktů, v pořádku: {len(valid)}.")
    return len(valid) == len(meta_files)

if __name__ == "__main__":
    validate_all_dumps_cli()