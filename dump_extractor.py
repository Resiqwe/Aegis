import json
import time
from pathlib import Path

# Cílový adresář pro naše surové dumpy (vždy vedle skriptu, ne podle toho, odkud se spouští)
DUMP_DIR = Path(__file__).resolve().parent / "dumps"
DUMP_DIR.mkdir(parents=True, exist_ok=True)

def extract_and_dump(neuron_core, run_id: int):
    """
    Sklidí reálná data a telemetrii z běžící instance neuronu
    a bezpečně je uloží na disk s jednotným kódováním.
    Váhy se ukládají jako čísla (jedna váha na řádek), bias a pravděpodobnosti jdou do meta JSONu.
    """
    timestamp = int(time.time())
    base_name = f"dump_run_{run_id}_{timestamp}"

    # 1. Sběr telemetrie z reálného objektu
    telemetry = {
        "run_id": run_id,
        "timestamp": timestamp,
        "epochs": getattr(neuron_core, "epochs", 0),
        "final_loss": getattr(neuron_core, "loss", 0.0),
        "bias": getattr(neuron_core, "bias", 0.0),
        "probabilities": list(getattr(neuron_core, "probabilities", [])),
        "status": "OK"
    }

    json_path = DUMP_DIR / f"{base_name}_meta.json"
    # Použití utf-8-sig pro plnou konzistenci při čtení
    with open(json_path, "w", encoding="utf-8-sig") as f:
        json.dump(telemetry, f, indent=4, ensure_ascii=False)

    # 2. Sběr reálných vah z vnitřní matice/atributu jádra
    weights = getattr(neuron_core, "weights", [])
    weights_path = DUMP_DIR / f"{base_name}_weights.dat"

    with open(weights_path, "w", encoding="utf-8-sig") as f:
        for w in weights:
            f.write(f"{float(w)!r}\n")

    print(f"[DUMP EXTRACTOR] Reálný dump úspěšně uložen: {DUMP_DIR.name}/{base_name}_*")
    return json_path
