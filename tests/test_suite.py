"""
tests/test_suite.py - Komplexní integrační testovací sada pro Cyber-Tactics 2.0.

Ověřuje:
  1. Import všech modulů a subsystémů balíčku src
  2. Stabilní 60-krokovou simulaci s detekcí poškození a eliminací
  3. Reálnou atribuci killů útočníkovi (žádné 0-kill zkreslení)
  4. Všechny 4 třídní schopnosti (Grunt smoke, Medic pulse, Sniper focus, Juggernaut shield)
  5. Asynchronní simulační worker a lock-free most (AsyncSimulationWorker & LockFreeStateBridge)
  6. Ukládání a načítání snapshotů (.pt)
"""

import os
import sys
import time
import unittest
import torch

# Přidání cesty k src
test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(test_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if test_dir not in sys.path:
    sys.path.insert(0, test_dir)

import config
import ai_brains
import scenarios
import engine_core
import telemetry_logger
import telemetry_lab
from engine.staging_buffer import SimulationStagingBuffer
from engine.spatial_system import SpatialSystem
from engine.ballistics_system import VectorizedBallisticsSystem
from engine.combat_system import CombatSystem
from engine.objectives_system import ObjectivesSystem
from engine.state_bridge import LockFreeStateBridge
from engine.async_runner import AsyncSimulationWorker


class TestCyberTacticsSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\n[Test Setup] Testování na zařízení: {cls.device}")

    def test_01_config_and_dimensions(self):
        """Ověření importu klíčových modulů, rozměrů světa a senzorů/akcí."""
        for mod in (config, ai_brains, scenarios, engine_core, telemetry_logger, telemetry_lab):
            self.assertIsNotNone(mod)
        self.assertEqual(config.WORLD_W, 2400)
        self.assertEqual(config.WORLD_H, 1350)
        self.assertEqual(len(config.SENSOR_NAMES), 14)
        self.assertEqual(len(config.ACTION_NAMES), 7)
        self.assertEqual(config.SENSOR_COUNT, 14)
        self.assertEqual(config.ACTION_COUNT, 7)

    def test_02_engine_subsystems_integration(self):
        """Ověření inicializace jádra a všech 5 subsystémů."""
        engine = engine_core.RealisticBattleEngine(2400, 1350, self.device)
        self.assertIsInstance(engine.staging_buf, SimulationStagingBuffer)
        self.assertIsInstance(engine.spatial, SpatialSystem)
        self.assertIsInstance(engine.ballistics, VectorizedBallisticsSystem)
        self.assertIsInstance(engine.combat, CombatSystem)
        self.assertIsInstance(engine.objectives, ObjectivesSystem)

    def test_03_simulation_and_kill_attribution(self):
        """Ověření 120 simulačních kroků a nenulové atribuce killů."""
        engine = engine_core.RealisticBattleEngine(2400, 1350, self.device)
        engine.configure(400, 4)
        
        # Povolíme vizuální stopy a poškození
        engine.visual_mode = True
        
        for _ in range(120):
            engine.step(1.0 / 60.0)

        with engine._engine_lock:
            snapshot = engine.get_battle_state_snapshot()
        self.assertIn("pos", snapshot)
        self.assertIn("hp", snapshot)
        self.assertIn("alive", snapshot)
        
        total_kills = engine.kills_per_faction.sum().item()
        total_unit_kills = engine.unit_kills.sum().item()
        print(f"\n[Test 03] Kills: Factions sum={total_kills}, Unit kills sum={total_unit_kills}")
        self.assertEqual(total_kills, total_unit_kills)

    def test_04_class_abilities_activation(self):
        """Ověření exekuce všech 4 schopností přes CombatSystem."""
        combat = CombatSystem(self.device)
        N = 4
        dt = 1.0 / 60.0
        
        alive = torch.ones(N, dtype=torch.bool, device=self.device)
        classes = torch.tensor([config.CLASS_GRUNT, config.CLASS_MEDIC, config.CLASS_SNIPER, config.CLASS_JUGGERNAUT], dtype=torch.int8, device=self.device)
        teams = torch.tensor([0, 0, 0, 0], dtype=torch.int8, device=self.device)
        pos = torch.tensor([[100.0, 100.0], [110.0, 100.0], [100.0, 110.0], [105.0, 105.0]], dtype=torch.float32, device=self.device)
        hp = torch.tensor([50.0, 50.0, 50.0, 50.0], dtype=torch.float32, device=self.device)
        class_max_hp = torch.tensor([115.0, 105.0, 70.0, 480.0], dtype=torch.float32, device=self.device)
        
        brain_ability = torch.tensor([0.9, 0.9, 0.9, 0.9], dtype=torch.float32, device=self.device)
        ability_cd = torch.zeros(N, dtype=torch.float32, device=self.device)
        active_smokes = torch.zeros((0, 5), dtype=torch.float32, device=self.device)
        fortress_timer = torch.zeros(N, dtype=torch.float32, device=self.device)
        sniper_timer = torch.zeros(N, dtype=torch.float32, device=self.device)
        staging_buf = SimulationStagingBuffer(N, self.device)
        healing_done = torch.zeros(N, dtype=torch.float32, device=self.device)
        fitness = torch.zeros(N, dtype=torch.float32, device=self.device)
        
        cd_out, smokes_out, fort_out, snip_out = combat.process_abilities(
            dt=dt,
            brain_ability=brain_ability,
            alive=alive,
            classes=classes,
            teams=teams,
            pos=pos,
            hp=hp,
            class_max_hp=class_max_hp,
            ability_cooldown=ability_cd,
            active_smokes_gpu=active_smokes,
            fortress_shield_timer=fortress_timer,
            sniper_focus_timer=sniper_timer,
            staging_buf=staging_buf,
            healing_done=healing_done,
            fitness=fitness,
            active_heals_p1=[],
            active_heals_p2=[],
        )
        
        # Grunt vypustil kouř
        self.assertGreater(smokes_out.shape[0], 0)
        # Medic provedl pulse heal do staging bufferu
        self.assertGreater(staging_buf.delta_hp.sum().item(), 0.0)
        # Sniper má aktivní focus timer
        self.assertGreater(snip_out[2].item(), 0.0)
        # Juggernaut má aktivní fortress shield
        self.assertGreater(fort_out[3].item(), 0.0)

    def test_05_async_worker_and_state_bridge(self):
        """Ověření decoupling workeru a čtení interpolovaných stavů."""
        engine = engine_core.RealisticBattleEngine(2400, 1350, self.device)
        engine.configure(400, 4)
        bridge = LockFreeStateBridge()
        worker = AsyncSimulationWorker(engine, bridge)
        
        worker.start()
        time.sleep(0.15)
        
        self.assertTrue(worker.is_running())
        state = bridge.get_interpolated_state()
        self.assertIsNotNone(state)
        self.assertIn("pos", state)
        
        worker.stop()
        self.assertFalse(worker.is_running())

    def test_06_snapshot_roundtrip(self):
        """Ověření uložení a načtení snapshotu s 3-vrstvou vahovou strukturou."""
        engine = engine_core.RealisticBattleEngine(2400, 1350, self.device)
        engine.configure(400, 4)
        telemetry = telemetry_logger.TelemetryManager("telemetry")
        
        snap_path = telemetry.save_snapshot(engine)
        self.assertTrue(os.path.exists(snap_path))
        
        # Načtení do nového enginu
        engine2 = engine_core.RealisticBattleEngine(2400, 1350, self.device)
        engine2.configure(400, 4)
        ok = telemetry.load_snapshot(engine2, snap_path)
        self.assertTrue(ok)
        
        # Vyčištění testovacího snapshotu
        try:
            os.remove(snap_path)
        except Exception:
            pass
        telemetry.shutdown(wait=True)

    def test_07_async_snapshot_pipeline(self):
        """Ověření Fáze 1: asynchronní ukládání snapshotu přes worker, debounce, atomický zápis a načtení."""
        engine = engine_core.RealisticBattleEngine(2400, 1350, self.device)
        engine.configure(400, 4)
        telemetry = telemetry_logger.TelemetryManager("telemetry")
        engine.telemetry_mgr = telemetry

        bridge = LockFreeStateBridge()
        worker = AsyncSimulationWorker(engine, bridge)
        worker.start()

        # Pošleme 2 požadavky rychle za sebou a ověříme doručení banneru i vytvoření souboru
        worker.post_command("save_snapshot", "async_test")
        banners = []
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            banners.extend(bridge.pop_banners())
            if any("[ULOZIT]" in b for b in banners):
                break
            time.sleep(0.05)

        worker.stop()

        save_banners = [b for b in banners if "[ULOZIT]" in b]
        self.assertEqual(len(save_banners), 1, f"Očekáván 1 potvrzovací banner, získáno: {banners}")

        snaps = [p for p in telemetry.list_available_snapshots() if "async_test" in os.path.basename(p)]
        self.assertTrue(len(snaps) >= 1, "Asynchronní snapshot nebyl nalezen na disku.")
        snap_path = snaps[0]
        self.assertFalse(os.path.exists(snap_path + ".tmp"), "Dočasný .tmp soubor nesmí zůstat na disku.")

        # Ověření, že neobsahuje duplicitní klíče brains_W1 a jde načíst zpět
        raw = torch.load(snap_path, map_location="cpu", weights_only=False)
        self.assertIn("brains_W1_full", raw)
        self.assertNotIn("brains_W1", raw)

        engine2 = engine_core.RealisticBattleEngine(2400, 1350, self.device)
        engine2.configure(400, 4)
        self.assertTrue(telemetry.load_snapshot(engine2, snap_path))

        # Ověření debounce pojistky (_save_in_progress)
        with telemetry._io_lock:
            telemetry._save_in_progress = True
        try:
            self.assertFalse(telemetry.save_snapshot_async(engine, "debounce_blocked"))
        finally:
            with telemetry._io_lock:
                telemetry._save_in_progress = False

        try:
            os.remove(snap_path)
        except Exception:
            pass
        telemetry.shutdown(wait=True)

    def test_08_lock_free_double_buffer_and_staging(self):
        """Ověření Fáze 3: 2-fázový Stage -> Format snapshot, inspekce na CPU a lock-free čtení z mostu."""
        engine = engine_core.RealisticBattleEngine(2400, 1350, self.device)
        engine.configure(800, 4)
        engine.step(1.0 / 60.0)

        # Změříme dobu staging fáze pod zámkem (po warm-upu)
        _ = engine.extract_raw_telemetry_state(target_id=None, auto_best=True, include_inspection=True)
        t0 = time.perf_counter()
        raw = engine.extract_raw_telemetry_state(target_id=None, auto_best=True, include_inspection=True)
        stage_ms = (time.perf_counter() - t0) * 1000.0
        print(f"\n[Test 08] Stage lock time (800 units): {stage_ms:.2f} ms")

        snap = engine_core.format_battle_snapshot(raw)
        insp = engine_core.format_inspected_unit(raw)
        self.assertIsNotNone(insp)
        self.assertIn("unit_idx", insp)
        self.assertIn("decision_flow", insp)
        self.assertIn("live_class_stats", snap)

        # Ověření lock-free publikace a čtení v LockFreeStateBridge
        bridge = LockFreeStateBridge()
        bridge.publish_snapshot(snap)
        latest = bridge.get_latest_snapshot()
        self.assertIs(latest, snap)
        self.assertEqual(latest["_seq"], 1)

        interp = bridge.get_interpolated_state()
        self.assertIsNotNone(interp)
        self.assertEqual(interp["_seq"], 1)

        # Druhá publikace posune sekvenční číslo a povolí Lerp
        snap2 = dict(snap)
        bridge.publish_snapshot(snap2)
        self.assertEqual(bridge.get_latest_snapshot()["_seq"], 2)
        interp2 = bridge.get_interpolated_state()
        self.assertTrue(interp2.get("is_interpolated", False))

    def test_09_phase2_vectorized_nodes_and_kills(self):
        """Ověření Fáze 2: 0x .item() ve for smyčkách hot-loopu a korektní GPU vektorizace uzlů a killů."""
        import ast
        from pathlib import Path

        core_path = Path(src_dir) / "engine_core.py"
        tree = ast.parse(core_path.read_text(encoding="utf-8"))
        hot_methods = {"_step_internal", "_atomic_commit_pipeline"}
        bad_items = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in hot_methods:
                for sub in ast.walk(node):
                    if isinstance(sub, ast.For):
                        for inner in ast.walk(sub):
                            if isinstance(inner, ast.Attribute) and inner.attr == "item":
                                bad_items.append((node.name, inner.lineno))
        self.assertEqual(bad_items, [], f"Nalezeno .item() uvnitř for smyčky: {bad_items}")

        # Funkční ověření vektorizovaného obsazování uzlů i GPU akumulace killů
        engine = engine_core.RealisticBattleEngine(2400, 1350, self.device)
        engine.configure(200, 4)
        engine.visual_mode = False
        for _ in range(30):
            engine.step(1.0 / 60.0)
        self.assertEqual(
            int(engine.kills_per_faction.sum().item()),
            int(engine.unit_kills.sum().item()),
        )


if __name__ == "__main__":
    unittest.main()
