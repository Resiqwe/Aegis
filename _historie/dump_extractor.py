import hashlib
import json
import time
from pathlib import Path

# Cílový adresář pro naše surové dumpy (vždy vedle skriptu)
DUMP_DIR = Path(__file__).resolve().parent / "dumps"
DUMP_DIR.mkdir(parents=True, exist_ok=True)

def extract_and_dump(neuron_core, run_id: int):
    """
    Sklidí reálná data a telemetrii z běžící instance neuronu,
    spočítá SHA-256 otisk souboru vah a bezpečně je uloží na disk.
    """
    timestamp = int(time.time())
    base_name = f"dump_run_{run_id}_{timestamp}"

    # 1. Sběr reálných vah z vnitřní matice/atributu jádra
    weights = getattr(neuron_core, "weights", [])
    weights_path = DUMP_DIR / f"{base_name}_weights.dat"

    # Zápis vah do souboru (otevřený a editovatelný formát)
    with open(weights_path, "w", encoding="utf-8-sig") as f:
        for w in weights:
            f.write(f"{float(w)!r}\n")

    # Výpočet SHA-256 haše souboru vah (kryptografická pečeť)
    with open(weights_path, "rb") as wf:
        weights_bytes = wf.read()
        weights_hash = hashlib.sha256(weights_bytes).hexdigest()

    # 2. Sběr telemetrie včetně haše vah do metadat
    telemetry = {
        "run_id": run_id,
        "timestamp": timestamp,
        "epochs": getattr(neuron_core, "epochs", 0),
        "final_loss": getattr(neuron_core, "loss", 0.0),
        "bias": getattr(neuron_core, "bias", 0.0),
        "probabilities": list(getattr(neuron_core, "probabilities", [])),
        "weights_hash": weights_hash,  # <--- Zde je uložena pečeť integrity
        "status": "OK"
    }

    json_path = DUMP_DIR / f"{base_name}_meta.json"
    with open(json_path, "w", encoding="utf-8-sig") as f:
        json.dump(telemetry, f, indent=4, ensure_ascii=False)

    print(f"[DUMP EXTRACTOR] Reálný dump úspěšně uložen (se zapečetěným hashem): {DUMP_DIR.name}/{base_name}_*")
    return json_path