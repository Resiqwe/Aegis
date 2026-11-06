r"""
Jedna spoušť na celou trubku: trénink -> dump -> validace -> analýza -> zpětná kontrola.
Když kterýkoli krok selže, řetěz se zastaví a řekne kde (exit code 1).

Spuštění:  .\.venv\Scripts\python.exe run_all.py
"""
import json
import sys
from pathlib import Path

import numpy as np

import pipeline
import dump_validator
import dump_analyzer
from neuron_core import NeuronCore, X_env, y_env

HERE = Path(__file__).resolve().parent
POCET_BEHU = 3


def krok(nazev):
    print("\n" + "#" * 60)
    print(f"# {nazev}")
    print("#" * 60)


def zpetna_kontrola(meta_files):
    """Načte uložené váhy z disku, spustí predikci a porovná s tím, co jádro při běhu vypočítalo."""
    for meta_path in meta_files:
        with open(meta_path, "r", encoding="utf-8-sig") as f:
            meta = json.load(f)
        weights_path = meta_path.with_name(meta_path.name.replace("_meta.json", "_weights.dat"))
        with open(weights_path, "r", encoding="utf-8-sig") as f:
            W = np.array([float(x) for x in f.read().split()])

        core = NeuronCore(input_dim=len(W))
        core.W, core.b = W, float(meta["bias"])
        pravdepodobnosti, rozhodnuti = core.predict(X_env)

        if not np.allclose(pravdepodobnosti, meta["probabilities"], atol=1e-9):
            raise RuntimeError(f"{meta_path.name}: predikce z uložených vah se liší od predikce při běhu.")
        if not np.array_equal(rozhodnuti, y_env):
            raise RuntimeError(f"{meta_path.name}: rozhodnutí {rozhodnuti} neodpovídá cíli {y_env}.")
        print(f"  OK {meta_path.name}: uložené váhy dávají stejné výsledky jako běh a trefují cíl")


def main():
    try:
        krok("1/4 PIPELINE (trénink + dump)")
        pipeline.main(POCET_BEHU)

        krok("2/4 VALIDÁTOR")
        vsechny = sorted(dump_validator.DUMP_DIR.glob("*_meta.json"))
        platne = dump_validator.get_valid_dumps(quiet=False)
        print(f"Zkontrolováno {len(vsechny)}, v pořádku {len(platne)}.")
        if len(platne) != len(vsechny):
            raise RuntimeError("Ve složce dumps/ jsou neplatné artefakty (viz výše). Řetěz zastaven.")

        krok("3/4 ANALÝZA")
        dump_analyzer.analyze_dumps()

        krok("4/4 ZPĚTNÁ KONTROLA (načti váhy z disku a ověř)")
        zpetna_kontrola(platne)

    except (pipeline.PipelineError, RuntimeError) as e:
        print(f"\n[CHYBA] {e}")
        sys.exit(1)

    print("\n=== TRUBKA TEČE: všechny kroky proběhly a výsledky sedí ===")


if __name__ == "__main__":
    main()
