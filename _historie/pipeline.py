import json
import os
import subprocess
import sys
import time
from pathlib import Path

from dump_extractor import extract_and_dump

HERE = Path(__file__).resolve().parent


class PipelineError(RuntimeError):
    """Krok pipeline selhal. Zastaví celý řetěz s jasnou zprávou."""


def spust_beh_jara(run_id):
    """Spustí jedno čisté kolo neuron_core.py, převezme jeho stav a pošle ho do dumperu."""
    start = time.time()
    # Spustíme skript jako proces a počkáme na dokončení
    # Dítě musí psát v UTF-8, jinak ho na Windows (cp1250) rodič nepřečte
    prostredi = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    sleva = subprocess.run(
        [sys.executable, str(HERE / "neuron_core.py")],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=HERE, env=prostredi
    )
    trvani = time.time() - start

    if sleva.returncode != 0:
        raise PipelineError(f"Běh #{run_id}: jádro skončilo s kódem {sleva.returncode}:\n{sleva.stderr.strip()}")

    # Strojově čitelný stav (řádek STATE_JSON z jádra), žádné hledání v textu výpisu
    radky = [r for r in (sleva.stdout or "").splitlines() if r.startswith("STATE_JSON: ")]
    if not radky:
        raise PipelineError(f"Běh #{run_id}: jádro nevypsalo STATE_JSON. Nic se neuložilo.")
    try:
        stav = json.loads(radky[-1][len("STATE_JSON: "):])
    except json.JSONDecodeError as e:
        raise PipelineError(f"Běh #{run_id}: STATE_JSON je poškozený ({e}).")

    class LiveNeuronSnapshot:
        pass

    snapshot = LiveNeuronSnapshot()
    snapshot.epochs = stav["epochs"]
    snapshot.loss = stav["loss"]
    snapshot.weights = stav["weights"]
    snapshot.bias = stav["bias"]
    snapshot.probabilities = stav["probabilities"]

    # Uložení reálného dumpu na disk
    extract_and_dump(snapshot, run_id=run_id)

    return {
        "status": "OK",
        "trvani": trvani,
        "epochy": stav["epochs"],
        "loss": stav["loss"],
        "raw": sleva.stdout,
    }


def main(pocet_behu=3):
    print("=== [PIPELINE ZAVADĚČ] Spouštím orchestraci laboratoře ===")
    zaznamy = []

    print(f"Plán: Provedu {pocet_behu} po sobě jdoucí čisté běhy jádra...\n")

    for i in range(1, pocet_behu + 1):
        print(f"Spouštím běh #{i}...", end="", flush=True)
        vysledek = spust_beh_jara(run_id=i)
        zaznamy.append(vysledek)
        print(f" Hotovo ({vysledek['trvani']:.2f}s) | Epochy: {vysledek['epochy']} | Loss: {vysledek['loss']:.6f}")

    print("\n" + "=" * 50)
    print(f"=== SOUHRN PIPELINE ({pocet_behu} běhů dokončeno) ===")
    print("=" * 50)
    for i, z in enumerate(zaznamy, 1):
        print(f"Běh #{i} -> Status: {z['status']} | Čas: {z['trvani']:.2f}s | Epochy: {z['epochy']} | Poslední Loss: {z['loss']:.6f}")
    print("=" * 50)
    print("Všechny běhy proběhly bez zásahu do samotného core.")
    return zaznamy


if __name__ == "__main__":
    try:
        main()
    except PipelineError as e:
        print(f"\n[CHYBA PIPELINE] {e}")
        sys.exit(1)
