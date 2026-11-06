import os
import stat
import time
import json
import hashlib
from pathlib import Path

import numpy as np

# ==============================================================================
# 0. DISKOVÁ HYGIENA A SEPARACE ZÓN (Pilíř 3: dumps/ vs. logs/)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DUMP_DIR = BASE_DIR / "dumps"          # Výhradně stavové artefakty (.dat, _meta.json, HEAD.json)
LOG_DIR = BASE_DIR / "logs"            # Výhradně kontinuální auditní proud (node_events.jsonl)
DUMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

HEAD_POINTER_PATH = DUMP_DIR / "HEAD.json"
EVENT_LOG_PATH = LOG_DIR / "node_events.jsonl"


def _set_readonly(path: Path, readonly: bool = True):
    """Zamkne soubor na Read-Only (WORM pro běžné procesy) nebo jej odemkne pro autorizovaný admin zásah."""
    if not path.exists():
        return
    mode = stat.S_IREAD if readonly else (stat.S_IREAD | stat.S_IWRITE)
    os.chmod(path, mode)


def _atomic_write_bytes(target_path: Path, data: bytes, lock_readonly: bool = False):
    """
    Zapíše data nejprve do .tmp souboru a pak je atomicky vymění (os.replace).
    Pokud je cílový soubor zamčený (Read-Only), autorizovaně jej před výměnou odemkne
    a po zápisu případně znovu uzamkne (WORM).
    """
    tmp_path = target_path.with_suffix(target_path.suffix + f".{os.getpid()}.tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
    if target_path.exists():
        _set_readonly(target_path, readonly=False)
    os.replace(tmp_path, target_path)
    if lock_readonly:
        _set_readonly(target_path, readonly=True)


def append_audit_event(event_type: str, payload: dict):
    """
    Pilíř 2: Kontinuální append-only auditní stopa (logs/node_events.jsonl)
    s vlastním kryptografickým řetězem prev_event_hash -> event_hash.
    Zaznamenává jak úspěšné takty (TICK_OK), tak odražené sabotáže (SECURITY_REJECT).
    """
    prev_event_hash = "GENESIS_00000000000000000000000000000000000000000000000000000000"
    if EVENT_LOG_PATH.exists() and EVENT_LOG_PATH.stat().st_size > 0:
        with open(EVENT_LOG_PATH, "rb") as f:
            lines = f.read().splitlines()
            if lines:
                try:
                    last_obj = json.loads(lines[-1].decode("utf-8"))
                    prev_event_hash = last_obj.get("event_hash", prev_event_hash)
                except Exception:
                    prev_event_hash = "CORRUPTED_PREV_LINE"

    record = {
        "timestamp_us": time.time_ns() // 1_000,
        "event_type": event_type,
        "prev_event_hash": prev_event_hash,
        **payload,
    }
    canonical = json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")
    record["event_hash"] = hashlib.sha256(canonical).hexdigest()

    with open(EVENT_LOG_PATH, "a", encoding="utf-8") as log_f:
        log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


# ==============================================================================
# 1. JÁDRO (CORE) - Zrychlený cyber-neuron (BCE + Batch Clock + Warm-Start)
# ==============================================================================
class NeuronCore:
    def __init__(self, input_dim=3, learning_rate=0.5, seed=42):
        self.input_dim = input_dim
        np.random.seed(seed)
        self.W = np.random.randn(input_dim)
        self.b = 0.0
        self.lr = learning_rate
        self.total_epochs = 0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def load_from_verified_dump(self, meta_path: Path | None = None):
        """
        Warm-start: pokud není meta_path zadán, načte automaticky poslední platnou hlavu z HEAD.json.
        Po kontrole občanky (SHA-256) natáhne váhy i bias ze souboru _weights.dat.
        """
        if meta_path is None:
            if not HEAD_POINTER_PATH.exists():
                raise RuntimeError("[WARM-START] HEAD.json neexistuje, není na co navázat!")
            head_info = json.loads(HEAD_POINTER_PATH.read_text(encoding="utf-8-sig"))
            meta_path = DUMP_DIR / head_info["head_meta_file"]

        valid = validate_dumps([meta_path], quiet=True)
        if not valid:
            raise RuntimeError(f"[WARM-START] Dump {meta_path.name} neprošel kontrolou občanky!")

        with open(meta_path, "r", encoding="utf-8-sig") as f:
            meta = json.load(f)

        weights_path = DUMP_DIR / meta_path.name.replace("_meta.json", "_weights.dat")
        with open(weights_path, "r", encoding="utf-8-sig") as f:
            params = [float(x) for x in f.read().split()]

        if len(params) != self.input_dim + 1:
            msg = f"[WARM-START] Očekáváno {self.input_dim + 1} parametrů (W + bias), nalezeno {len(params)}."
            append_audit_event("SECURITY_REJECT", {"target": meta_path.name, "reason": msg})
            raise RuntimeError(msg)

        self.W = np.array(params[:-1], dtype=np.float64)
        self.b = float(params[-1])
        self.total_epochs = int(meta.get("total_epochs", meta.get("epochs", 0)))

    def train_fast(self, X, y, max_cas_sekund=2.0, target_loss=1e-4, batch_epochs=250):
        """
        Blesková tréninková smyčka s dvojitou závorou:
          1. Časový jistič (max_cas_sekund) - chrání před nekonečným while loopem
          2. Early-exit (target_loss)       - vypne okamžitě, jakmile je úloha vyřešená
        Používá Binary Cross-Entropy gradient (bez brzdy sigmoidy) a kontroluje hodiny jen po dávkách.
        """
        inv_n = 1.0 / len(X)
        lr_scaled = self.lr * inv_n
        X_T = X.T

        start = time.perf_counter()
        epochs_run = 0
        stop_reason = "TIME_LIMIT"

        y_pred = self._sigmoid(np.dot(X, self.W) + self.b)
        mse_loss = float(np.mean((y_pred - y) ** 2))

        while True:
            elapsed = time.perf_counter() - start
            if mse_loss <= target_loss:
                stop_reason = "EARLY_EXIT"
                break
            if elapsed >= max_cas_sekund:
                stop_reason = "TIME_LIMIT"
                break

            for _ in range(batch_epochs):
                y_pred = 1.0 / (1.0 + np.exp(-(np.dot(X, self.W) + self.b)))
                error = y_pred - y
                self.W -= lr_scaled * np.dot(X_T, error)
                self.b -= lr_scaled * float(np.sum(error))

            epochs_run += batch_epochs
            mse_loss = float(np.mean((y_pred - y) ** 2))

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.total_epochs += epochs_run
        y_pred = self._sigmoid(np.dot(X, self.W) + self.b)
        mse_loss = float(np.mean((y_pred - y) ** 2))
        bce_loss = float(-np.mean(y * np.log(y_pred + 1e-15) + (1.0 - y) * np.log(1.0 - y_pred + 1e-15)))

        return {
            "epochs_run": epochs_run,
            "total_epochs": self.total_epochs,
            "elapsed_ms": elapsed_ms,
            "mse_loss": mse_loss,
            "bce_loss": bce_loss,
            "stop_reason": stop_reason,
            "probabilities": [float(p) for p in y_pred],
        }

    def predict(self, X):
        """Vrátí pravděpodobnosti a rozhodnutí (0 nebo 1)."""
        z = np.dot(X, self.W) + self.b
        pravdepodobnosti = self._sigmoid(z)
        return pravdepodobnosti, np.round(pravdepodobnosti)


# Data (prostředí buňky)
X_env = np.array([
    [1.0, 2.0, 1.0],
    [2.0, 1.0, 2.0],
    [3.0, 2.0, 0.0],
    [4.0, 3.0, 1.0]
])
y_env = np.array([0.0, 0.0, 1.0, 1.0])


# ==============================================================================
# 2. EXTRAKTOR - Odemčené váhy (.dat) + Zamčená WORM občanka (_meta.json) + HEAD
# ==============================================================================
def extract_and_dump(
    core: NeuronCore,
    stats: dict,
    run_id: int,
    parent_hash: str | None = None,
    silent: bool = False,
):
    """
    - _weights.dat zůstává odemčený (Read-Write) pro rychlou práci, ale chráněný hashem.
    - _meta.json (občanka) se po atomickém zápisu zamkne jako Read-Only (WORM).
    - HEAD.json atomicky ukazuje na nejnovější platný dump.
    - Událost TICK_OK se zapíše do odděleného logs/node_events.jsonl.
    """
    timestamp_us = time.time_ns() // 1_000
    base_name = f"dump_run_{run_id}_{timestamp_us}"

    all_params = [float(w) for w in core.W] + [float(core.b)]
    weights_text = "".join(f"{p!r}\n" for p in all_params)
    weights_bytes = weights_text.encode("utf-8-sig")
    weights_hash = hashlib.sha256(weights_bytes).hexdigest()

    weights_path = DUMP_DIR / f"{base_name}_weights.dat"
    _atomic_write_bytes(weights_path, weights_bytes, lock_readonly=False)

    telemetry = {
        "run_id": run_id,
        "timestamp_us": timestamp_us,
        "param_count": len(all_params),
        "epochs": stats["epochs_run"],
        "total_epochs": stats["total_epochs"],
        "elapsed_ms": round(stats["elapsed_ms"], 3),
        "stop_reason": stats["stop_reason"],
        "final_loss": stats["mse_loss"],
        "bce_loss": stats["bce_loss"],
        "probabilities": stats["probabilities"],
        "parent_weights_hash": parent_hash,
        "weights_hash": weights_hash,
        "status": "OK",
    }

    json_path = DUMP_DIR / f"{base_name}_meta.json"
    json_bytes = json.dumps(telemetry, indent=4, ensure_ascii=False).encode("utf-8-sig")
    # Občanka je po vystavení uzamčena jako Read-Only (odemknout ji smí jen admin/controller)
    _atomic_write_bytes(json_path, json_bytes, lock_readonly=True)

    # Aktualizace ukazatele HEAD.json na nejnovější stav
    head_payload = {
        "updated_us": timestamp_us,
        "run_id": run_id,
        "head_meta_file": json_path.name,
        "head_weights_file": weights_path.name,
        "weights_hash": weights_hash,
        "total_epochs": stats["total_epochs"],
        "final_loss": stats["mse_loss"],
    }
    _atomic_write_bytes(
        HEAD_POINTER_PATH,
        json.dumps(head_payload, indent=4, ensure_ascii=False).encode("utf-8-sig"),
        lock_readonly=True,
    )

    append_audit_event("TICK_OK", telemetry)

    if not silent:
        render_telemetry_row(telemetry)

    return json_path, weights_hash


def authorize_admin_override(meta_path: Path, new_weights_data: list[float], reason: str = "ADMIN_OVERRIDE"):
    """
    Výjimka z WORM (Bod 4): Autorizovaný nadřazený proces / admin má právo
    odemknout zamčenou občanku, upravit váhy, přepočítat pečeť a zapsat auditní záznam.
    """
    _set_readonly(meta_path, readonly=False)
    with open(meta_path, "r", encoding="utf-8-sig") as f:
        meta = json.load(f)

    old_hash = meta.get("weights_hash")
    weights_path = DUMP_DIR / meta_path.name.replace("_meta.json", "_weights.dat")

    weights_text = "".join(f"{float(p)!r}\n" for p in new_weights_data)
    weights_bytes = weights_text.encode("utf-8-sig")
    new_hash = hashlib.sha256(weights_bytes).hexdigest()
    _atomic_write_bytes(weights_path, weights_bytes, lock_readonly=False)

    # Přepočet predikce pro autorizované váhy
    core = NeuronCore(input_dim=len(new_weights_data) - 1)
    core.W = np.array(new_weights_data[:-1], dtype=np.float64)
    core.b = float(new_weights_data[-1])
    probs, _ = core.predict(X_env)

    meta["param_count"] = len(new_weights_data)
    meta["parent_weights_hash"] = old_hash
    meta["weights_hash"] = new_hash
    meta["probabilities"] = [float(p) for p in probs]
    meta["status"] = reason

    json_bytes = json.dumps(meta, indent=4, ensure_ascii=False).encode("utf-8-sig")
    _atomic_write_bytes(meta_path, json_bytes, lock_readonly=True)

    append_audit_event("ADMIN_OVERRIDE", {
        "meta_file": meta_path.name,
        "old_hash": old_hash,
        "new_hash": new_hash,
        "reason": reason,
    })
    return new_hash


# ==============================================================================
# 3. VALIDÁTOR - Kontrola integrity, občanky a zápis SECURITY_REJECT do logu
# ==============================================================================
def validate_dumps(meta_files, quiet: bool = False) -> list:
    """Zkontroluje strukturu, počet parametrů (W+b), platnost čísel a SHA-256."""
    valid = []
    for meta_path in meta_files:
        base_prefix = meta_path.name.replace("_meta.json", "")
        weights_path = DUMP_DIR / f"{base_prefix}_weights.dat"

        if not weights_path.exists():
            msg = f"[VALIDÁTOR] Chybí soubor vah k metadatům: {base_prefix}"
            append_audit_event("SECURITY_REJECT", {"target": base_prefix, "reason": msg})
            raise RuntimeError(msg)

        try:
            with open(meta_path, "r", encoding="utf-8-sig") as f:
                meta = json.load(f)
        except Exception as e:
            msg = f"[VALIDÁTOR] Poškozený JSON u {base_prefix}: {e}"
            append_audit_event("SECURITY_REJECT", {"target": base_prefix, "reason": msg})
            raise RuntimeError(msg)

        raw_bytes = weights_path.read_bytes()
        current_hash = hashlib.sha256(raw_bytes).hexdigest()
        expected_hash = meta.get("weights_hash")
        if expected_hash and current_hash != expected_hash:
            msg = f"[VALIDÁTOR] Narušení integrity! {weights_path.name} byl změněn (haš nesouhlasí)."
            append_audit_event("SECURITY_REJECT", {"target": weights_path.name, "reason": msg})
            raise RuntimeError(msg)

        try:
            hodnoty = raw_bytes.decode("utf-8-sig").split()
            if not hodnoty:
                raise ValueError("žádné hodnoty")
            parsed = [float(h) for h in hodnoty]
        except Exception as e:
            msg = f"[VALIDÁTOR] Váhy v {weights_path.name} nejsou platná čísla: {e}"
            append_audit_event("SECURITY_REJECT", {"target": weights_path.name, "reason": msg})
            raise RuntimeError(msg)

        expected_count = meta.get("param_count")
        if expected_count is not None and len(parsed) != expected_count:
            msg = f"[VALIDÁTOR] Nesedí počet parametrů v {weights_path.name}: {len(parsed)} != {expected_count}"
            append_audit_event("SECURITY_REJECT", {"target": weights_path.name, "reason": msg})
            raise RuntimeError(msg)

        valid.append(meta_path)

    if not quiet:
        print(f"  [VALIDÁTOR] Ověřeno {len(valid)}/{len(meta_files)} občanek (SHA-256 + WORM Read-Only OK).")
    return valid


# ==============================================================================
# 4. ZPĚTNÁ KONTROLA - načti z disku a ověř
# ==============================================================================
def roundtrip_verification(valid_meta_files, quiet: bool = False):
    """Načte W i bias čistě z .dat souboru, spustí predikci a porovná ji s občankou i cílem."""
    for meta_path in valid_meta_files:
        with open(meta_path, "r", encoding="utf-8-sig") as f:
            meta = json.load(f)
        weights_path = DUMP_DIR / meta_path.name.replace("_meta.json", "_weights.dat")
        with open(weights_path, "r", encoding="utf-8-sig") as f:
            params = [float(x) for x in f.read().split()]

        core = NeuronCore(input_dim=len(params) - 1)
        core.W = np.array(params[:-1], dtype=np.float64)
        core.b = float(params[-1])
        pravdepodobnosti, rozhodnuti = core.predict(X_env)

        if not np.allclose(pravdepodobnosti, meta["probabilities"], atol=1e-9):
            msg = f"{meta_path.name}: predikce z uložených vah se liší od predikce při běhu."
            append_audit_event("SECURITY_REJECT", {"target": meta_path.name, "reason": msg})
            raise RuntimeError(msg)
        if not np.array_equal(rozhodnuti, y_env):
            msg = f"{meta_path.name}: rozhodnutí {rozhodnuti} neodpovídá cíli {y_env}."
            append_audit_event("SECURITY_REJECT", {"target": meta_path.name, "reason": msg})
            raise RuntimeError(msg)

    if not quiet:
        print(f"  [ROUND-TRIP] Všech {len(valid_meta_files)} modelů (.dat) dává bitově shodnou predikci s občankou.")


# ==============================================================================
# 5. PILÍŘ 1: SCADA TERMINÁLOVÝ DASHBOARD (Glanceability místo print chaosu)
# ==============================================================================
TABLE_HEADER = (
    "  ┌──────┬────────┬────────────┬──────────┬─────────────────┬────────────┬───────────────────────────┐\n"
    "  │ TAKT │ STATUS │ STOP DŮVOD │  LATENCE │ EPOCHY (CELKEM) │  MSE LOSS  │ RODOKMEN (RODIČ -> NOVÝ)  │\n"
    "  ├──────┼────────┼────────────┼──────────┼─────────────────┼────────────┼───────────────────────────┤"
)
TABLE_FOOTER = (
    "  └──────┴────────┴────────────┴──────────┴─────────────────┴────────────┴───────────────────────────┘"
)


def render_telemetry_header():
    print(TABLE_HEADER)


def render_telemetry_row(t: dict):
    parent_short = (t["parent_weights_hash"][:8] if t.get("parent_weights_hash") else "GENESIS ")
    curr_short = t["weights_hash"][:8]
    ep_str = f"+{t['epochs']:<4} ({t['total_epochs']:>5})"
    print(
        f"  │ #{t['run_id']:<3} │ {t['status']:<6} │ {t['stop_reason']:<10} │ "
        f"{t['elapsed_ms']:>5.2f} ms │ {ep_str:<15} │ {t['final_loss']:>10.2e} │ "
        f"{parent_short} -> {curr_short}   │"
    )


def render_telemetry_footer():
    print(TABLE_FOOTER)


# ==============================================================================
# ORCHESTRACE HLAVNÍHO BĚHU
# ==============================================================================
def run_pipeline(silent: bool = False):
    t_celkem = time.perf_counter()
    if not silent:
        print("=" * 98)
        print("  NEURON NODE | SCADA TELEMETRIE | WORM OBČANKY (dumps/) | AUDITNÍ PROUD (logs/node_events.jsonl)")
        print("=" * 98)
        render_telemetry_header()

    prahy_loss = [1e-4, 2.2e-5, 5e-6]

    vytvorene = []
    posledni_meta = None
    posledni_hash = None

    for run_id, target_loss in enumerate(prahy_loss, start=1):
        core = NeuronCore(input_dim=3, learning_rate=0.5)
        if posledni_meta is not None:
            core.load_from_verified_dump(posledni_meta)

        stats = core.train_fast(
            X_env,
            y_env,
            max_cas_sekund=2.0,
            target_loss=target_loss,
            batch_epochs=250,
        )
        meta_path, w_hash = extract_and_dump(
            core, stats, run_id, parent_hash=posledni_hash, silent=silent
        )
        vytvorene.append(meta_path)
        posledni_meta = meta_path
        posledni_hash = w_hash

    if not silent:
        render_telemetry_footer()

    platne = validate_dumps(vytvorene, quiet=silent)
    roundtrip_verification(platne, quiet=silent)

    celkem_ms = (time.perf_counter() - t_celkem) * 1000.0
    if not silent:
        print(f"  [HOTOVO] 3 takty + WORM pečetění + verifikace za {celkem_ms:.2f} ms | HEAD -> {HEAD_POINTER_PATH.name}")
        print("=" * 98)


if __name__ == "__main__":
    try:
        run_pipeline()
    except RuntimeError as e:
        print(f"\n[CHYBA] {e}")
        raise SystemExit(1)
