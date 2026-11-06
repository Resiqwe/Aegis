r"""
Standalone zátěžová, kouřová a sabotážní sada pro neuron_node.py.
Ověřuje i 3 pilíře kultury logování a správy stavu:
  - Pilíř 1: SCADA tabulkový přehled + tichý režim (silent=True) bez brzdění terminálem
  - Pilíř 2: Kontinuální append-only logs/node_events.jsonl s prev_event_hash řetězem a záznamy SECURITY_REJECT
  - Pilíř 3: Separace dumps/ vs. logs/, ukazatel HEAD.json, WORM Read-Only občanky a autorizovaný Admin Override

Spuštění:
  .\.venv\Scripts\python.exe -X utf8 test_torture_suite.py
"""
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from neuron import neuron_node as nn

BASE_DIR = Path(__file__).resolve().parent.parent
DUMP_DIR = nn.DUMP_DIR
LOG_DIR = nn.LOG_DIR
BACKUP_DUMPS = BASE_DIR / "_dumps_backup_during_test"
BACKUP_LOGS = BASE_DIR / "_logs_backup_during_test"


def _unlock_all_in_dir(d: Path):
    if d.exists():
        for p in d.rglob("*"):
            if p.is_file():
                try:
                    os.chmod(p, stat.S_IREAD | stat.S_IWRITE)
                except OSError:
                    pass


def clean_zones():
    """Odemkne a vyčistí testovací zóny dumps/ a logs/."""
    for d in (DUMP_DIR, LOG_DIR):
        _unlock_all_in_dir(d)
        if d.exists():
            for p in d.iterdir():
                if p.is_file():
                    p.unlink()
        else:
            d.mkdir(parents=True, exist_ok=True)


class TestRunner:
    def __init__(self):
        self.results = []

    def run_positive(self, code: str, name: str, fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            ms = (time.perf_counter() - t0) * 1000.0
            self.results.append((code, name, "PASS", ms, detail or "OK"))
            print(f"  ✅ [{code}] {name:<56} | PASS ({ms:6.2f} ms) | {detail or 'OK'}")
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000.0
            self.results.append((code, name, "FAIL", ms, f"Neočekávaná chyba: {e}"))
            print(f"  ❌ [{code}] {name:<56} | FAIL ({ms:6.2f} ms) | {e}")

    def run_sabotage(self, code: str, name: str, fn, expected_exc=RuntimeError):
        t0 = time.perf_counter()
        try:
            fn()
            ms = (time.perf_counter() - t0) * 1000.0
            msg = "SABOTÁŽ PROŠLA BEZ POVŠIMNUTÍ (buňka nevyhodila výjimku)!"
            self.results.append((code, name, "FAIL", ms, msg))
            print(f"  ❌ [{code}] {name:<56} | FAIL ({ms:6.2f} ms) | {msg}")
        except expected_exc as e:
            ms = (time.perf_counter() - t0) * 1000.0
            short_err = str(e).splitlines()[0]
            self.results.append((code, name, "PASS", ms, f"Útok odražen: {short_err}"))
            print(f"  🛡️  [{code}] {name:<56} | PASS ({ms:6.2f} ms) | Zachyceno: {short_err}")
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000.0
            self.results.append((code, name, "FAIL", ms, f"Jiná výjimka ({type(e).__name__}): {e}"))
            print(f"  ❌ [{code}] {name:<56} | FAIL ({ms:6.2f} ms) | Jiná výjimka: {type(e).__name__}: {e}")


# ==============================================================================
# 1. KLASICKÉ KOUŘÁKY + SEPARACE ZÓN (S0-S6)
# ==============================================================================
def smoke_s0_subprocess_cli():
    clean_zones()
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", str(BASE_DIR / "neuron" / "neuron_node.py")],
        capture_output=True, text=True, encoding="utf-8", cwd=BASE_DIR
    )
    assert proc.returncode == 0, f"Subprocess selhal ({proc.returncode}): {proc.stderr}"
    metas = sorted(DUMP_DIR.glob("dump_run_*_meta.json"))
    dats = sorted(DUMP_DIR.glob("dump_run_*_weights.dat"))
    assert len(metas) == 3 and len(dats) == 3
    assert nn.HEAD_POINTER_PATH.exists(), "Chybí HEAD.json v dumps/"
    assert nn.EVENT_LOG_PATH.exists(), "Chybí node_events.jsonl v logs/"
    assert not list(DUMP_DIR.glob("*.jsonl")), "Porušena hygiena: .jsonl nesmí být v dumps/!"
    assert "│ TAKT │ STATUS │" in proc.stdout, "Chybí SCADA tabulka ve výstupu!"
    return "Subprocess exit 0, dumps/ a logs/ odděleny, SCADA tabulka OK"


def smoke_s1():
    clean_zones()
    core = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    stats = core.train_fast(nn.X_env, nn.y_env, max_cas_sekund=2.0, target_loss=1e-4, batch_epochs=250)
    assert stats["stop_reason"] == "EARLY_EXIT"
    assert stats["mse_loss"] <= 1e-4
    return f"EARLY_EXIT za {stats['elapsed_ms']:.2f} ms, loss={stats['mse_loss']:.2e}"


def smoke_s2():
    clean_zones()
    core = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    stats = core.train_fast(nn.X_env, nn.y_env, target_loss=1e-4)
    meta_path, _ = nn.extract_and_dump(core, stats, run_id=1, silent=True)

    weights_path = DUMP_DIR / meta_path.name.replace("_meta.json", "_weights.dat")
    assert meta_path.exists() and weights_path.exists() and nn.HEAD_POINTER_PATH.exists()
    assert not list(DUMP_DIR.glob("*.tmp"))
    lines = weights_path.read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 4
    return "Soubory .dat (4 param), WORM _meta.json, HEAD.json i logs/ vytvořeny"


def smoke_s3():
    meta_files = sorted(DUMP_DIR.glob("dump_run_*_meta.json"))
    nn.validate_dumps(meta_files, quiet=True)
    nn.roundtrip_verification(meta_files, quiet=True)
    return "Predikce z disku sedí na 1e-9 s během i cílem [0,0,1,1]"


def smoke_s4_head_warmstart():
    # Warm-start bez zadání cesty -> automaticky použije HEAD.json!
    prvnimeta = sorted(DUMP_DIR.glob("dump_run_*_meta.json"))[0]
    prvni_hash = json.loads(prvnimeta.read_text(encoding="utf-8-sig"))["weights_hash"]

    core2 = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    core2.load_from_verified_dump()  # None -> čte přímo HEAD.json
    stats2 = core2.train_fast(nn.X_env, nn.y_env, target_loss=5e-6)
    meta2_path, _ = nn.extract_and_dump(core2, stats2, run_id=2, parent_hash=prvni_hash, silent=True)

    meta2 = json.loads(meta2_path.read_text(encoding="utf-8-sig"))
    assert meta2["parent_weights_hash"] == prvni_hash
    assert meta2["total_epochs"] == 1750
    return f"Automatický Warm-Start přes HEAD.json OK (total_epochs={meta2['total_epochs']})"


def smoke_s5_determinism():
    clean_zones()
    c1 = nn.NeuronCore(input_dim=3, learning_rate=0.5, seed=42)
    _, h1 = nn.extract_and_dump(c1, c1.train_fast(nn.X_env, nn.y_env, target_loss=1e-4), run_id=1, silent=True)
    c2 = nn.NeuronCore(input_dim=3, learning_rate=0.5, seed=42)
    _, h2 = nn.extract_and_dump(c2, c2.train_fast(nn.X_env, nn.y_env, target_loss=1e-4), run_id=2, silent=True)
    assert h1 == h2
    return f"Bitová shoda SHA-256 ({h1[:10]}...)"


def smoke_s6_admin_override():
    clean_zones()
    core = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    meta_path, old_hash = nn.extract_and_dump(
        core, core.train_fast(nn.X_env, nn.y_env, target_loss=1e-4), run_id=1, silent=True
    )
    # Autorizovaný admin proces odemkne občanku, jemně upraví váhy a znovu ji zapečetí
    new_params = [float(w) for w in core.W] + [float(core.b) + 0.001]
    new_hash = nn.authorize_admin_override(meta_path, new_params, reason="ADMIN_CALIBRATION")
    assert new_hash != old_hash
    nn.validate_dumps([meta_path], quiet=True)
    nn.roundtrip_verification([meta_path], quiet=True)
    # Kontrola, že občanka je po zásahu admina opět zamčená (Read-Only)
    assert not (meta_path.stat().st_mode & stat.S_IWRITE), "Občanka nezůstala po admin zásahu zamčená!"
    return "Autorizovaný Admin Override přepečetil občanku a zapsal auditní stopu"


# ==============================================================================
# 2. ZÁTĚŽOVÉ TESTY (STRESS & ENDURANCE Z1-Z3)
# ==============================================================================
def stress_z1_burst():
    clean_zones()
    N = 50
    vytvorene = []
    posledni_hash = None

    for i in range(1, N + 1):
        core = nn.NeuronCore(input_dim=3, learning_rate=0.5)
        if i > 1:
            core.load_from_verified_dump()  # přes HEAD.json
        stats = core.train_fast(nn.X_env, nn.y_env, max_cas_sekund=0.05, target_loss=1e-4 / i, batch_epochs=50)
        meta_path, w_hash = nn.extract_and_dump(core, stats, run_id=i, parent_hash=posledni_hash, silent=True)
        vytvorene.append(meta_path)
        posledni_hash = w_hash

    assert len(set(vytvorene)) == N
    nn.validate_dumps(vytvorene, quiet=True)
    nn.roundtrip_verification(vytvorene, quiet=True)

    # Kontrola dvojitého řetězu: parent_weights_hash I prev_event_hash v logs/node_events.jsonl
    events = [json.loads(line) for line in nn.EVENT_LOG_PATH.read_text(encoding="utf-8").splitlines()]
    assert len(events) == N
    for idx in range(1, N):
        assert events[idx]["parent_weights_hash"] == events[idx - 1]["weights_hash"]
        assert events[idx]["prev_event_hash"] == events[idx - 1]["event_hash"]

    return "50 taktů (silent=True), 50/50 občanek i dvojitý hash-chain v logs/ OK"


def stress_z2_xor_watchdog():
    X_xor = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    y_xor = np.array([0.0, 1.0, 1.0, 0.0])
    core = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    stats = core.train_fast(X_xor, y_xor, max_cas_sekund=0.025, target_loss=1e-6, batch_epochs=100)
    assert stats["stop_reason"] == "TIME_LIMIT"
    assert 20.0 <= stats["elapsed_ms"] <= 60.0
    return f"XOR nezamrzl: vypnul přes TIME_LIMIT za {stats['elapsed_ms']:.2f} ms ({stats['epochs_run']} epoch)"


def stress_z3_reality_inversion():
    clean_zones()
    core1 = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    nn.extract_and_dump(core1, core1.train_fast(nn.X_env, nn.y_env, target_loss=1e-4), run_id=1, silent=True)

    y_inv = np.array([1.0, 1.0, 0.0, 0.0])
    core2 = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    core2.load_from_verified_dump()
    s2 = core2.train_fast(nn.X_env, y_inv, max_cas_sekund=0.5, target_loss=1e-4)
    _, rozhodnuti = core2.predict(nn.X_env)
    assert np.array_equal(rozhodnuti, y_inv)
    return f"Přeučeno z [0,0,1,1] na [1,1,0,0] za {s2['elapsed_ms']:.2f} ms (+{s2['epochs_run']} epoch)"


# ==============================================================================
# 3. SABOTÁŽ ZVENČÍ (EXTERNAL TAMPERING E1-E9)
# ==============================================================================
def _create_clean_dump():
    clean_zones()
    core = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    stats = core.train_fast(nn.X_env, nn.y_env, target_loss=1e-4)
    meta_path, _ = nn.extract_and_dump(core, stats, run_id=1, silent=True)
    weights_path = DUMP_DIR / meta_path.name.replace("_meta.json", "_weights.dat")
    return meta_path, weights_path


def sabotage_e1_tamper_weight():
    meta_path, weights_path = _create_clean_dump()
    lines = weights_path.read_text(encoding="utf-8-sig").splitlines()
    lines[0] = repr(float(lines[0]) + 1e-6)
    weights_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    nn.validate_dumps([meta_path], quiet=True)


def sabotage_e2_tamper_bias():
    meta_path, weights_path = _create_clean_dump()
    lines = weights_path.read_text(encoding="utf-8-sig").splitlines()
    lines[-1] = repr(float(lines[-1]) - 1e-12)
    weights_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    victim = nn.NeuronCore(input_dim=3)
    victim.load_from_verified_dump(meta_path)


def sabotage_e3_worm_direct_meta_overwrite():
    meta_path, _ = _create_clean_dump()
    # Běžný neautorizovaný proces zkusí přímo přepsat zamčenou WORM občanku -> OS vyhodí PermissionError!
    with open(meta_path, "w", encoding="utf-8-sig") as f:
        f.write('{"hacked": true}')


def sabotage_e4_rehashed_garbage():
    meta_path, weights_path = _create_clean_dump()
    evil_bytes = "3.14\nHACKED_NEURON\n-1.0\n0.5\n".encode("utf-8-sig")
    weights_path.write_bytes(evil_bytes)
    nn._set_readonly(meta_path, readonly=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    meta["weights_hash"] = hashlib.sha256(evil_bytes).hexdigest()
    meta_path.write_text(json.dumps(meta), encoding="utf-8-sig")
    nn.validate_dumps([meta_path], quiet=True)


def sabotage_e5_rehashed_missing_bias():
    meta_path, weights_path = _create_clean_dump()
    lines = weights_path.read_text(encoding="utf-8-sig").splitlines()[:3]
    evil_bytes = ("\n".join(lines) + "\n").encode("utf-8-sig")
    weights_path.write_bytes(evil_bytes)
    nn._set_readonly(meta_path, readonly=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    meta["weights_hash"] = hashlib.sha256(evil_bytes).hexdigest()
    meta_path.write_text(json.dumps(meta), encoding="utf-8-sig")
    nn.validate_dumps([meta_path], quiet=True)


def sabotage_e6_deleted_weights():
    meta_path, weights_path = _create_clean_dump()
    weights_path.unlink()
    nn.validate_dumps([meta_path], quiet=True)


def sabotage_e7_corrupted_json():
    meta_path, _ = _create_clean_dump()
    nn._set_readonly(meta_path, readonly=False)
    meta_path.write_text("{corrupted_json: true, ", encoding="utf-8-sig")
    nn.validate_dumps([meta_path], quiet=True)


def sabotage_e8_spoofed_probabilities_in_meta():
    meta_path, _ = _create_clean_dump()
    nn._set_readonly(meta_path, readonly=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
    meta["probabilities"][0] += 0.05
    meta_path.write_text(json.dumps(meta), encoding="utf-8-sig")
    nn.roundtrip_verification([meta_path], quiet=True)


def sabotage_e9_swap_attack():
    clean_zones()
    c1 = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    m1, _ = nn.extract_and_dump(c1, c1.train_fast(nn.X_env, nn.y_env, target_loss=1e-3), run_id=1, silent=True)
    c2 = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    m2, _ = nn.extract_and_dump(c2, c2.train_fast(nn.X_env, nn.y_env, target_loss=1e-5), run_id=2, silent=True)

    w1 = DUMP_DIR / m1.name.replace("_meta.json", "_weights.dat")
    w2 = DUMP_DIR / m2.name.replace("_meta.json", "_weights.dat")
    w2.write_bytes(w1.read_bytes())
    nn.validate_dumps([m2], quiet=True)


# ==============================================================================
# 4. SABOTÁŽ ZEVNITŘ (INTERNAL POISONING I1-I4)
# ==============================================================================
def sabotage_i1_nan_weights_poisoning():
    clean_zones()
    core = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    stats = core.train_fast(nn.X_env, nn.y_env, target_loss=1e-4)
    core.W[0] = float("nan")
    meta_path, _ = nn.extract_and_dump(core, stats, run_id=1, silent=True)
    nn.roundtrip_verification([meta_path], quiet=True)


def sabotage_i2_dimension_mismatch_warmstart():
    meta_path, _ = _create_clean_dump()
    wrong_core = nn.NeuronCore(input_dim=5, learning_rate=0.5)
    wrong_core.load_from_verified_dump(meta_path)


def internal_i3_extreme_input_saturation():
    X_extreme = np.array([
        [-1000.0, -1000.0, 1000.0],
        [-500.0, -500.0, 500.0],
        [1000.0, 1000.0, -1000.0],
        [500.0, 500.0, -500.0],
    ])
    y_target = np.array([0.0, 0.0, 1.0, 1.0])
    core = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = core.train_fast(X_extreme, y_target, max_cas_sekund=0.05, target_loss=1e-4, batch_epochs=50)
        probs, _ = core.predict(X_extreme)
    assert not np.any(np.isnan(probs)) and not np.any(np.isnan(core.W))
    return f"Přežilo extrémní vstupy ±1000 bez NaN (loss={stats['mse_loss']:.2e})"


def internal_i4_ram_protection_and_audit_trail():
    meta_path, weights_path = _create_clean_dump()
    weights_path.write_text("0.0\n0.0\n0.0\n0.0\n", encoding="utf-8-sig")

    victim = nn.NeuronCore(input_dim=3, learning_rate=0.5)
    w_before = victim.W.copy()
    try:
        victim.load_from_verified_dump(meta_path)
    except RuntimeError:
        assert np.array_equal(victim.W, w_before)
        # Ověříme, že útok byl zapsán do logs/node_events.jsonl jako SECURITY_REJECT!
        events = [json.loads(line) for line in nn.EVENT_LOG_PATH.read_text(encoding="utf-8").splitlines()]
        assert events[-1]["event_type"] == "SECURITY_REJECT"
        return "RAM nedotčena + útok zapsán do logs/node_events.jsonl jako SECURITY_REJECT"
    raise AssertionError("Warm-start načetl otrávený dump!")


# ==============================================================================
# HLAVNÍ ORCHESTRACE TESTŮ
# ==============================================================================
def main():
    for b in (BACKUP_DUMPS, BACKUP_LOGS):
        if b.exists():
            _unlock_all_in_dir(b)
            shutil.rmtree(b)
    if DUMP_DIR.exists():
        shutil.copytree(DUMP_DIR, BACKUP_DUMPS)
    if LOG_DIR.exists():
        shutil.copytree(LOG_DIR, BACKUP_LOGS)

    t_start = time.perf_counter()
    runner = TestRunner()

    print("=" * 112)
    print("=== STANDALONE TORTURE & SABOTAGE SUITE PRO NEURON_NODE.PY (23 TESTŮ + WORM + AUDIT CHAIN + SEPARACE ZÓN) ===")
    print("=" * 112)

    try:
        print("\n[1/4] KLASICKÉ KOUŘÁKY & KULTURA STAVU (Smoke Tests S0-S6)")
        runner.run_positive("S0", "Subprocess CLI + SCADA tabulka + separace dumps/ a logs/", smoke_s0_subprocess_cli)
        runner.run_positive("S1", "Studený start a Early-Exit pod 50 ms", smoke_s1)
        runner.run_positive("S2", "Tvorba artefaktů (.dat, WORM _meta.json, HEAD.json, logs/)", smoke_s2)
        runner.run_positive("S3", "Zpětná kontrola (Round-trip z disku)", smoke_s3)
        runner.run_positive("S4", "Automatický Warm-Start přes ukazatel HEAD.json", smoke_s4_head_warmstart)
        runner.run_positive("S5", "Bitový determinismus při stejném seedu", smoke_s5_determinism)
        runner.run_positive("S6", "Autorizovaný Admin Override zamčené WORM občanky", smoke_s6_admin_override)

        print("\n[2/4] ZÁTĚŽOVÉ TESTY (Stress & Endurance Z1-Z3)")
        runner.run_positive("Z1", "Rychlopalba 50 taktů (silent=True) + dvojitý hash-chain", stress_z1_burst)
        runner.run_positive("Z2", "Watchdog pod neřešitelným XOR (ochrana while loopu)", stress_z2_xor_watchdog)
        runner.run_positive("Z3", "Šoková inverze reality za chodu ([0,0,1,1]->[1,1,0,0])", stress_z3_reality_inversion)

        print("\n[3/4] SABOTÁŽ ZVENČÍ (External Disk Tampering E1-E9)")
        runner.run_sabotage("E1", "Falšování váhy W1 o 0.000001 v .dat", sabotage_e1_tamper_weight)
        runner.run_sabotage("E2", "Falšování samotného Biasu o 1e-12 na 4. řádku .dat", sabotage_e2_tamper_bias)
        runner.run_sabotage("E3", "Neautorizovaný přepis WORM _meta.json (OS Read-Only zámek)", sabotage_e3_worm_direct_meta_overwrite, expected_exc=PermissionError)
        runner.run_sabotage("E4", "Podvržení textu 'HACKED' + přepočet SHA-256 v občance", sabotage_e4_rehashed_garbage)
        runner.run_sabotage("E5", "Uříznutí biasu (3 řádky místo 4) + přepočet SHA-256", sabotage_e5_rehashed_missing_bias)
        runner.run_sabotage("E6", "Smazání souboru _weights.dat (osiřelá občanka)", sabotage_e6_deleted_weights)
        runner.run_sabotage("E7", "Poškození syntaxe _meta.json", sabotage_e7_corrupted_json)
        runner.run_sabotage("E8", "Zfalšování probabilities v _meta.json (Round-trip past)", sabotage_e8_spoofed_probabilities_in_meta)
        runner.run_sabotage("E9", "Podvržení cizího mozku (Swap Attack mezi Během #1 a #2)", sabotage_e9_swap_attack)

        print("\n[4/4] SABOTÁŽ ZEVNITŘ (Internal Poisoning I1-I4)")
        runner.run_sabotage("I1", "Injekce NaN do vah před uložením", sabotage_i1_nan_weights_poisoning)
        runner.run_sabotage("I2", "Warm-start dumpu (dim=3) do buňky s jinou dimenzí (dim=5)", sabotage_i2_dimension_mismatch_warmstart)
        runner.run_positive("I3", "Extrémní vstupy ±1000 (odolnost proti přetečení exp)", internal_i3_extreme_input_saturation)
        runner.run_positive("I4", "Ochrana RAM + zápis SECURITY_REJECT do logs/node_events.jsonl", internal_i4_ram_protection_and_audit_trail)

    finally:
        clean_zones()
        if BACKUP_DUMPS.exists():
            shutil.rmtree(DUMP_DIR)
            shutil.move(str(BACKUP_DUMPS), str(DUMP_DIR))
        if BACKUP_LOGS.exists():
            shutil.rmtree(LOG_DIR)
            shutil.move(str(BACKUP_LOGS), str(LOG_DIR))
        # Pokud před testem byly zóny prázdné, vygenerujeme jeden čistý referenční běh
        if not list(DUMP_DIR.glob("dump_run_*_meta.json")):
            nn.run_pipeline(silent=True)

    total_ms = (time.perf_counter() - t_start) * 1000.0
    passed = sum(1 for r in runner.results if r[2] == "PASS")
    total = len(runner.results)

    print("\n" + "=" * 112)
    print(f"=== VÝSLEDEK SUITE: {passed}/{total} TESTŮ PROŠLO | ČAS: {total_ms:.1f} ms | Zóny dumps/ a logs/ čisté ===")
    print("=" * 112)

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
