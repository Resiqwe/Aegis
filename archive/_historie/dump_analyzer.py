import json
from pathlib import Path
from dump_validator import get_valid_dumps, DUMP_DIR

# DUMP_DIR se sdílí s validátorem (cesta vedle skriptu)

def analyze_dumps():
    """
    Proleze výhradně validované artefakty, takže se nemusí bát poškozených dat.
    """
    meta_files = get_valid_dumps(quiet=True)
    
    if not meta_files:
        print(f"[ANALYZÁTOR] Žádné platné a ověřené dumpy k analýze v {DUMP_DIR}.")
        return

    print("=" * 60)
    print(f"=== [ANALÝZA LATENTNÍHO PROSTORU] Ověřených dumpů: {len(meta_files)} ===")
    print("=" * 60)

    for meta_path in meta_files:
        # Bezpečné čtení s utf-8-sig
        try:
            with open(meta_path, "r", encoding="utf-8-sig") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[ANALYZÁTOR] Chyba při čtení {meta_path.name}: {e}")
            continue
            
        base_prefix = meta_path.name.replace("_meta.json", "")
        weights_path = DUMP_DIR / f"{base_prefix}_weights.dat"
        
        print(f"\n[Běh #{meta.get('run_id', '?')}] (Časová značka: {meta.get('timestamp')})")
        print(f"  -> Epochy: {meta.get('epochs')}")
        print(f"  -> Finální loss: {meta.get('final_loss')}")
        print(f"  -> Status: {meta.get('status')}")
        
        # Čtení vah
        with open(weights_path, "r", encoding="utf-8-sig") as wf:
            lines = wf.read().strip().splitlines()
            print(f"  -> Surový soubor vah: {weights_path.name} ({len(lines)} řádků dat)")

    print("=" * 60)
    print("Analýza dokončena bezpečně nad ověřenými daty.")

if __name__ == "__main__":
    analyze_dumps()