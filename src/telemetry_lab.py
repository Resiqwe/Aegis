"""
telemetry_lab.py - Samostatná Výzkumná a Analytická Laboratoř pro Cyber-Tactics AI
Stand-alone analytický modul pro hloubkové zpracování telemetrie, evoluce a chování AI:
  1. Meta-Game & Křivky Adaptace:
     - Vývoj nákupních preferencí generálů (třídy: Pěšák, Medik, Sniper, Juggernaut)
     - Vývoj Fitness a poměru vítězství jednotlivých frakcí
  2. Ekonomika & ROI Tříd:
     - Návratnost investice (Damage/bod, Léčení/bod, přežití)
  3. XAI (Explainable AI) & Analýza Neuronových Sítí:
     - Feature Importance: Které ze 14 senzorů mají největší vliv na střelbu, pohyb a přebití
     - Genetická diverzita populace: Detekce stagnace nebo zamrznutí v lokálním minimu
  4. Prostorová Taktická Kartografie:
     - Rozložení killů a rizikových zón zvlášť pro Domination, Citadel a Chasm
  5. Výstupy:
     - Formátovaný přehled v terminálu
     - Multi-panelový grafický report (PNG) s vysokým rozlišením
     - Samostatný interaktivní HTML report
"""

import os
import sys
import glob
import json
import argparse
import math
from datetime import datetime

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import matplotlib
# Default to Agg backend so it runs headlessly without Tk window issues unless --gui is specified
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

FACTION_NAMES = ["ALPHA VANGUARD", "OMEGA SYNDICATE", "SOLAR CORE", "PHANTOM UNIT"]
FACTION_COLORS = ["#00F0D2", "#FF2D73", "#FFC319", "#AF50FA"]
CLASS_NAMES = ["PĚŠÁK", "MEDIK", "SNIPER", "JUGGERNAUT"]
CLASS_COLORS = ["#4A90E2", "#41EB64", "#FFC319", "#E74C3C"]

try:
    from ai_brains import SENSOR_NAMES, ACTION_NAMES
except Exception:
    SENSOR_NAMES = [
        "Cíl Vpřed/Vzad", "Cíl Vpravo/Vlevo", "Blízkost cíle",
        "Palba Vpřed/Vzad", "Palba Vpravo/Vlevo",
        "Povel Vpřed/Vzad", "Povel Vpravo/Vlevo",
        "Úroveň potlačení", "Stav zásobníku", "Stav zdraví (HP)",
        "Kryt Vpřed/Vzad", "Kryt Vpravo/Vlevo",
        "Koheze čety", "Stav zóny",
    ]
    ACTION_NAMES = ["Plyn (Vpřed/Vzad)", "Úkrok (Vpravo/Vlevo)", "Spoušť palby", "Přebít zbraň"]


class TelemetryLab:
    def __init__(self, session_filter: str = "", telemetry_dir: str = "telemetry"):
        self.telemetry_dir = telemetry_dir
        self.session_filter = session_filter
        self.session_file = None
        self.session_id = None
        self.epochs_data = []
        self.spatial_df = None
        self.latest_snapshot_path = None
        self.snapshot_data = None

        self._find_and_load_data()

    def _find_and_load_data(self):
        """Vyhledá a načte nejnovější nebo vybranou relaci a související snapshot."""
        if not os.path.exists(self.telemetry_dir):
            return

        # 1. Nalezení JSONL logu
        pattern = os.path.join(self.telemetry_dir, "telemetry_session_*.jsonl")
        files = glob.glob(pattern)
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)

        if self.session_filter:
            target = next((f for f in files if self.session_filter in f), None)
        else:
            # Najít nejnovější neprázdný log
            target = next((f for f in files if os.path.getsize(f) > 500), None)
            if not target and files:
                target = files[0]

        if not target or not os.path.exists(target):
            return

        self.session_file = target
        self.session_id = os.path.basename(target).replace("telemetry_session_", "").replace(".jsonl", "")

        # Načtení záznamů epoch
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if obj.get("type") == "EPOCH_COMPLETED":
                            self.epochs_data.append(obj)
                    except Exception:
                        pass

        # 2. Načtení prostorových dat (Parquet)
        cand_pq = os.path.join(self.telemetry_dir, f"spatial_events_{self.session_id}.parquet")
        if not os.path.exists(cand_pq):
            pqs = glob.glob(os.path.join(self.telemetry_dir, "spatial_events_*.parquet"))
            pqs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            cand_pq = pqs[0] if pqs else None

        if cand_pq and os.path.exists(cand_pq):
            try:
                self.spatial_df = pd.read_parquet(cand_pq)
            except Exception:
                self.spatial_df = None

        # 3. Načtení nejnovějšího snapshotu pro neuronovou analýzu
        snaps = glob.glob(os.path.join(self.telemetry_dir, "snapshot_*.pt"))
        snaps.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        if snaps:
            self.latest_snapshot_path = snaps[0]
            try:
                self.snapshot_data = torch.load(self.latest_snapshot_path, map_location="cpu", weights_only=False)
            except Exception:
                self.snapshot_data = None

    def analyze_neural_weights(self) -> dict:
        """
        XAI Analýza vah neuronových sítí:
        Vypočítá efektivní matici propojení mezi 14 senzory a 4 akcemi (Feature Importance)
        a míru genetické diverzity (variance vah v populaci).
        """
        if not self.snapshot_data:
            return {}

        W1 = self.snapshot_data.get("brains_W1_full")
        if W1 is None:
            W1 = self.snapshot_data.get("brains_W1")
        W2 = self.snapshot_data.get("brains_W2_full")
        if W2 is None:
            W2 = self.snapshot_data.get("brains_W2")

        if W1 is None or W2 is None:
            return {}

        # W1: (N, 10, 24), W2: (N, 24, 4)
        W1 = W1.float()
        W2 = W2.float()
        N = W1.shape[0]

        # Efektivní spojovací síla: Saliency(s, a) = mean_units( |W1[:, s, :] @ W2[:, :, a]| )
        # Matice (10, 4): významnost každého senzoru pro každou akci
        with torch.no_grad():
            eff_matrix = torch.bmm(torch.abs(W1), torch.abs(W2)) # (N, 10, 4)
            avg_importance = torch.mean(eff_matrix, dim=0).numpy() # (10, 4)

            # Celková důležitost senzorů přes všechny akce
            sensor_totals = np.sum(avg_importance, axis=1)
            sensor_rel_pct = (sensor_totals / (np.sum(sensor_totals) + 1e-6)) * 100.0

            # Genetická diverzita (rozptyl vah napříč populací)
            var_W1 = float(torch.var(W1).item())
            var_W2 = float(torch.var(W2).item())
            diversity_index = (var_W1 + var_W2) * 0.5

        return {
            "importance_matrix": avg_importance,
            "sensor_pct": sensor_rel_pct,
            "diversity_index": diversity_index,
            "units_analyzed": N,
        }

    def print_terminal_summary(self):
        """Vytiskne strukturovaný vědecký report v terminálu."""
        print("\n" + "=" * 84)
        print("   CYBER-TACTICS // AI ANALYTICAL LAB & RESEARCH ENGINE")
        print("=" * 84)

        if not self.epochs_data:
            print("   [!] Nebyla nalezena žádná data epoch k analýze.")
            print("=" * 84)
            return

        total_epochs = len(self.epochs_data)
        first_ep = self.epochs_data[0]
        last_ep = self.epochs_data[-1]
        trophies = last_ep.get("total_wins", [0, 0, 0, 0])

        print(f"   Relace:         {self.session_id}")
        print(f"   Odehráno generací: {total_epochs} (Epocha #{first_ep.get('epoch', 1)} až #{last_ep.get('epoch', total_epochs)})")
        if self.spatial_df is not None:
            print(f"   Bojových událostí: {len(self.spatial_df)} prostorových záznamů v Parquet")
        if self.latest_snapshot_path:
            print(f"   Nejnovější model:  {os.path.basename(self.latest_snapshot_path)}")

        print("\n" + "-" * 84)
        print("   1. META-STRATEGICKÝ STAV FRAKCÍ A VÝVOJ GENERÁLŮ:")
        print("-" * 84)
        f_stats = last_ep.get("factions", [])
        f_stats_0 = first_ep.get("factions", [])
        for i, fs in enumerate(f_stats):
            fname = FACTION_NAMES[i]
            lvl = fs.get("level", 0)
            xp = int(fs.get("xp", 0))
            fit_now = fs.get("fitness_avg", 0.0)
            fit_prev = f_stats_0[i].get("fitness_avg", 0.0) if i < len(f_stats_0) else fit_now
            delta_fit = fit_now - fit_prev
            wins = trophies[i] if i < len(trophies) else 0
            probs = fs.get("class_probs", [1.0, 0.0, 0.0, 0.0])
            p_str = f"Pěšák:{int(probs[0]*100):2d}% | Medik:{int(probs[1]*100):2d}% | Sniper:{int(probs[2]*100):2d}% | Jugg:{int(probs[3]*100):2d}%"
            print(f"   • {fname:<16}: L{lvl} ({xp:5d} XP) [W:{wins:2d}] | Fit: {fit_now:5.1f} (d {delta_fit:+5.1f})")
            print(f"     Naučená skladba:   [{p_str}]")

        # 2. ROI Analýza
        print("\n" + "-" * 84)
        print("   2. EKONOMIKA VOJENSKÉHO ROZPOČTU // ROI ANALÝZA TŘÍD:")
        print("-" * 84)
        print(f"   {'Třída':<12} {'Cena':<6} {'Nasazeno':<10} {'Prům.Fit':<11} {'ROI / 1 b':<11} {'Dmg / b':<11} {'Dmg Utrž.':<11} {'Přežití'}")
        print("   " + "-" * 84)
        
        # Agregace ROI přes všechny odehrané epochy v této session
        class_agg = {c: {"cost": [1, 2, 3, 4][c], "count": 0, "fit_sum": 0.0, "dmg_sum": 0.0, "dmg_taken_sum": 0.0, "heal_sum": 0.0, "surv_sum": 0.0, "n_samples": 0} for c in range(4)}
        for ep in self.epochs_data:
            for r in ep.get("class_roi", []):
                cid = r.get("class_id", 0)
                if cid in class_agg:
                    cnt = r.get("count", 0)
                    class_agg[cid]["count"] += cnt
                    class_agg[cid]["fit_sum"] += r.get("avg_fitness", 0.0) * cnt
                    class_agg[cid]["dmg_sum"] += r.get("total_damage", 0.0)
                    class_agg[cid]["dmg_taken_sum"] += r.get("total_damage_taken", 0.0)
                    class_agg[cid]["heal_sum"] += r.get("total_healing", 0.0)
                    class_agg[cid]["surv_sum"] += r.get("survival_rate", 0.0) * cnt
                    class_agg[cid]["n_samples"] += cnt

        for c in range(4):
            agg = class_agg[c]
            cname = CLASS_NAMES[c]
            cost = agg["cost"]
            tot_cnt = agg["count"]
            if tot_cnt > 0:
                avg_fit = agg["fit_sum"] / tot_cnt
                roi_pt = avg_fit / float(cost)
                dmg_per_pt = agg["dmg_sum"] / (tot_cnt * cost)
                tot_taken = agg["dmg_taken_sum"]
                surv_pct = (agg["surv_sum"] / tot_cnt) * 100.0
                print(f"   {cname:<12} {cost} b    {tot_cnt:<10d} {avg_fit:<11.1f} {roi_pt:<11.2f} {dmg_per_pt:<11.1f} {tot_taken:<11.0f} {surv_pct:5.1f} %")

        # 3. Neuronová analýza
        neural_res = self.analyze_neural_weights()
        if neural_res:
            print("\n" + "-" * 84)
            print("   3. EXPLAINABLE AI (XAI) // ATRIBUCE SENZORŮ & DIVERZITA MOZKŮ:")
            print("-" * 84)
            print(f"   Analyzováno neuronových sítí v snapshotu: {neural_res['units_analyzed']}")
            print(f"   Index genetické diverzity (variance vah): {neural_res['diversity_index']:.4f}", end=" ")
            if neural_res['diversity_index'] > 0.05:
                print("(Zdravá diverzita populace - probíhá aktivní evoluce)")
            else:
                print("(Nízká variance - populace konverguje k uniformitě)")

            print("\n   Relativní vliv senzorů na rozhodování vojáků (Feature Importance):")
            pcts = neural_res["sensor_pct"]
            for s_idx in range(len(SENSOR_NAMES)):
                bar = "#" * int(pcts[s_idx] * 0.8)
                print(f"   • {SENSOR_NAMES[s_idx]:<16}: {pcts[s_idx]:5.1f} %  {bar}")

        print("\n" + "=" * 84)

    def generate_visual_report(self, output_path: str = None) -> str:
        """
        Vygeneruje 6-panelový vědecký dashboard ve formátu PNG.
        """
        if not self.epochs_data:
            return None

        if not output_path:
            os.makedirs(self.telemetry_dir, exist_ok=True)
            output_path = os.path.join(self.telemetry_dir, f"ai_analysis_report_{self.session_id}.png")

        # Tmavý profesionální vědecký styl
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(18, 12), dpi=130)
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.28)

        fig.patch.set_facecolor("#0B0E14")

        # Titulek
        fig.suptitle(
            f"CYBER-TACTICS // AI NEURO-EVOLUTION & TELEMETRY LAB REPORT [Session: {self.session_id}]",
            fontsize=15, fontweight="bold", color="#00F0D2", y=0.98
        )

        epochs_nums = [ep.get("epoch", idx + 1) for idx, ep in enumerate(self.epochs_data)]

        # -------------------------------------------------------------
        # PANEL 1: Křivka učení AI (Průměrná Fitness v čase)
        # -------------------------------------------------------------
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.set_facecolor("#101520")
        for f in range(4):
            series = [ep.get("factions", [{}])[f].get("fitness_avg", 0.0) if f < len(ep.get("factions", [])) else 0.0 for ep in self.epochs_data]
            ax1.plot(epochs_nums, series, label=FACTION_NAMES[f], color=FACTION_COLORS[f], linewidth=2.0)
        ax1.set_title("Křivka Učení AI (Průměrná Fitness Frakcí)", fontsize=11, fontweight="bold", color="#EBF2FA")
        ax1.set_xlabel("Generace / Epocha", fontsize=9, color="#91A2BC")
        ax1.set_ylabel("Fitness Skóre", fontsize=9, color="#91A2BC")
        ax1.grid(True, linestyle="--", alpha=0.2, color="#445C80")
        ax1.legend(loc="upper left", framealpha=0.4, fontsize=8)

        # -------------------------------------------------------------
        # PANEL 2: Poměr vítězství (Win Share Pie Chart)
        # -------------------------------------------------------------
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor("#101520")
        wins = self.epochs_data[-1].get("total_wins", [0, 0, 0, 0])
        tot_wins = max(1, sum(wins))
        win_shares = [w / tot_wins for w in wins]
        wedges, texts, autotexts = ax2.pie(
            win_shares, labels=[f"Alpha\n({wins[0]})", f"Omega\n({wins[1]})", f"Solar\n({wins[2]})", f"Phantom\n({wins[3]})"],
            colors=FACTION_COLORS, autopct="%1.0f%%", startangle=140,
            textprops={"fontsize": 8, "color": "#EBF2FA"},
            wedgeprops={"edgecolor": "#101520", "linewidth": 1.5}
        )
        for at in autotexts:
            at.set_color("#0B0E14")
            at.set_weight("bold")
        ax2.set_title(f"Rozdělení Trofejí (Celkem {sum(wins)} výher)", fontsize=11, fontweight="bold", color="#EBF2FA")

        # -------------------------------------------------------------
        # PANEL 3: Meta-Strategická Adaptace (Vývoj nákupních vah tříd)
        # -------------------------------------------------------------
        ax3 = fig.add_subplot(gs[1, 0:2])
        ax3.set_facecolor("#101520")
        # Průměr vah přes všechny 4 generály v jednotlivých epochách
        class_trends = {c: [] for c in range(4)}
        for ep in self.epochs_data:
            probs_epoch = []
            for fs in ep.get("factions", []):
                probs_epoch.append(fs.get("class_probs", [1.0, 0.0, 0.0, 0.0]))
            if probs_epoch:
                avg_p = np.mean(probs_epoch, axis=0)
                for c in range(4):
                    class_trends[c].append(avg_p[c] * 100.0)

        for c in range(4):
            ax3.plot(epochs_nums, class_trends[c], label=CLASS_NAMES[c], color=CLASS_COLORS[c], linewidth=2.2)
        ax3.set_title("Evoluce Armádního Rozpočtu (Průměrné Preference Generálů v %)", fontsize=11, fontweight="bold", color="#EBF2FA")
        ax3.set_xlabel("Generace / Epocha", fontsize=9, color="#91A2BC")
        ax3.set_ylabel("Zastoupení v rozpočtu (%)", fontsize=9, color="#91A2BC")
        ax3.grid(True, linestyle="--", alpha=0.2, color="#445C80")
        ax3.legend(loc="center left", framealpha=0.4, fontsize=8)

        # -------------------------------------------------------------
        # PANEL 4: ROI Efektivita Tříd (Bar Chart)
        # -------------------------------------------------------------
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_facecolor("#101520")
        last_roi = self.epochs_data[-1].get("class_roi", [])
        if last_roi:
            c_names = [r.get("class_name", "") for r in last_roi]
            rois = [r.get("roi_per_point", 0.0) for r in last_roi]
            bars = ax4.bar(c_names, rois, color=CLASS_COLORS[:len(c_names)], edgecolor="#101520", width=0.55)
            for b in bars:
                h = b.get_height()
                ax4.text(b.get_x() + b.get_width() / 2., h + 1.0, f"{h:.1f}", ha="center", va="bottom", fontsize=8, color="#EBF2FA")
        ax4.set_title("ROI / 1 Bod Ceny (Nejnovější Epocha)", fontsize=11, fontweight="bold", color="#EBF2FA")
        ax4.set_ylabel("Fitness na 1 bod", fontsize=9, color="#91A2BC")
        ax4.grid(True, linestyle="--", alpha=0.2, color="#445C80")
        ax4.tick_params(axis="x", labelsize=8)

        # -------------------------------------------------------------
        # PANEL 5: XAI Senzorická Důležitost (Feature Importance)
        # -------------------------------------------------------------
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_facecolor("#101520")
        neural_res = self.analyze_neural_weights()
        if neural_res:
            pcts = neural_res["sensor_pct"]
            y_pos = np.arange(len(SENSOR_NAMES))
            ax5.barh(y_pos, pcts, color="#00F0D2", edgecolor="#101520", height=0.65)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(SENSOR_NAMES, fontsize=8, color="#EBF2FA")
            ax5.invert_yaxis()
            ax5.set_title(f"XAI: Vliv Senzorů na Rozhodování (Div: {neural_res['diversity_index']:.3f})", fontsize=10, fontweight="bold", color="#EBF2FA")
            ax5.set_xlabel("Relativní váha (%)", fontsize=8, color="#91A2BC")
            ax5.grid(True, linestyle="--", alpha=0.2, color="#445C80")
        else:
            ax5.text(0.5, 0.5, "Snapshot .pt není dostupný", ha="center", va="center", color="#91A2BC")
            ax5.set_title("XAI: Senzorická Důležitost", fontsize=10, fontweight="bold", color="#EBF2FA")

        # -------------------------------------------------------------
        # PANEL 6: Prostorová 2D Heatmapa Zabití (Kills Heatmap)
        # -------------------------------------------------------------
        ax6 = fig.add_subplot(gs[2, 1:3])
        ax6.set_facecolor("#101520")
        if self.spatial_df is not None and len(self.spatial_df) > 0:
            x_pts = self.spatial_df["x"].to_numpy()
            y_pts = self.spatial_df["y"].to_numpy()
            h2d, xedges, yedges = np.histogram2d(x_pts, y_pts, bins=[48, 28], range=[[0, 2400], [0, 1350]])
            im = ax6.imshow(
                h2d.T, origin="upper", extent=[0, 2400, 1350, 0],
                cmap="inferno", aspect="auto", interpolation="gaussian"
            )
            cbar = plt.colorbar(im, ax=ax6, fraction=0.035, pad=0.03)
            cbar.ax.tick_params(labelsize=7, colors="#91A2BC")
            cbar.set_label("Hustota eliminací", fontsize=8, color="#91A2BC")
            ax6.set_title(f"Prostorová 2D Heatmapa Bojiště ({len(self.spatial_df)} eliminací)", fontsize=11, fontweight="bold", color="#EBF2FA")
        else:
            ax6.text(0.5, 0.5, "Prostorová telemetrie zatím nemá dostatek bodů", ha="center", va="center", color="#91A2BC")
            ax6.set_title("Prostorová 2D Heatmapa Bojiště", fontsize=11, fontweight="bold", color="#EBF2FA")
        ax6.set_xlabel("Souřadnice X bojiště", fontsize=8, color="#91A2BC")
        ax6.set_ylabel("Souřadnice Y bojiště", fontsize=8, color="#91A2BC")

        plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        plt.close(fig)
        return output_path

    def generate_html_report(self, output_path: str = None) -> str:
        """
        Vygeneruje moderní samostatný HTML report pro prohlížeč.
        """
        if not self.epochs_data:
            return None

        if not output_path:
            os.makedirs(self.telemetry_dir, exist_ok=True)
            output_path = os.path.join(self.telemetry_dir, f"ai_report_{self.session_id}.html")

        # Ujistit se, že máme vygenerovaný obrázek pro vložení
        img_name = f"ai_analysis_report_{self.session_id}.png"
        img_path = os.path.join(self.telemetry_dir, img_name)
        if not os.path.exists(img_path):
            self.generate_visual_report(img_path)

        last_ep = self.epochs_data[-1]
        trophies = last_ep.get("total_wins", [0, 0, 0, 0])
        f_stats = last_ep.get("factions", [])

        # Sestavení HTML
        cards_html = ""
        for i in range(4):
            fname = FACTION_NAMES[i]
            col = FACTION_COLORS[i]
            fs = f_stats[i] if i < len(f_stats) else {}
            lvl = fs.get("level", 0)
            xp = int(fs.get("xp", 0))
            fit = fs.get("fitness_avg", 0.0)
            wins = trophies[i] if i < len(trophies) else 0
            probs = fs.get("class_probs", [1.0, 0.0, 0.0, 0.0])
            cards_html += f"""
            <div class="card" style="border-top: 3px solid {col};">
                <h3 style="color: {col}; margin: 0 0 8px 0;">{fname}</h3>
                <div class="stat"><span>Úroveň:</span> <strong>L{lvl} ({xp} XP)</strong></div>
                <div class="stat"><span>Výhry:</span> <strong>🏆 {wins}</strong></div>
                <div class="stat"><span>Prům. Fitness:</span> <strong style="color: #41EB64;">{fit:.1f}</strong></div>
                <div class="stat" style="margin-top: 8px; font-size: 11px; color: #91A2BC;">
                    Pěšák: {int(probs[0]*100)}% | Medik: {int(probs[1]*100)}%<br>
                    Sniper: {int(probs[2]*100)}% | Juggernaut: {int(probs[3]*100)}%
                </div>
            </div>
            """

        roi_rows = ""
        for r in last_ep.get("class_roi", []):
            roi_rows += f"""
            <tr>
                <td><strong>{r.get('class_name')}</strong></td>
                <td>{r.get('cost')} b</td>
                <td>{r.get('count')}</td>
                <td>{r.get('avg_fitness', 0.0):.1f}</td>
                <td><strong style="color: #41EB64;">{r.get('roi_per_point', 0.0):.2f}</strong></td>
                <td>{r.get('total_damage', 0.0):.0f}</td>
                <td>{r.get('total_healing', 0.0):.0f}</td>
                <td>{r.get('survival_rate', 0.0)*100:.1f} %</td>
            </tr>
            """

        battles_rows = ""
        for ep in reversed(self.epochs_data[-8:]):
            w_team = ep.get("winner_team", -1)
            w_name = FACTION_NAMES[w_team] if 0 <= w_team < 4 else "REMIZA"
            w_col = FACTION_COLORS[w_team] if 0 <= w_team < 4 else "#91A2BC"
            battles_rows += f"""
            <tr>
                <td>#{ep.get('epoch', 0):02d}</td>
                <td>{ep.get('scenario', 'N/A').split('(')[0].strip()}</td>
                <td>{ep.get('duration_sec', 0.0):.1f} s</td>
                <td style="color: {w_col}; font-weight: bold;">{w_name}</td>
                <td style="color: #91A2BC;">{ep.get('end_reason', '')}</td>
            </tr>
            """

        # 5. Hardwarová telemetrie a profilace nároků
        hw_files = glob.glob(os.path.join(self.telemetry_dir, "hardware_profiling_*.parquet"))
        hw_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        hw_html = ""
        if hw_files:
            try:
                hw_df = pd.read_parquet(hw_files[0])
                avg_sys_cpu = hw_df["sys_cpu_pct"].mean()
                avg_proc_cpu = hw_df["proc_cpu_pct"].mean()
                avg_sys_ram = hw_df["sys_ram_used_gb"].mean()
                avg_proc_ram = hw_df["proc_ram_mb"].mean()
                avg_gpu = hw_df["gpu_util_pct"].mean()
                avg_temp = hw_df["gpu_temp_c"].mean()
                avg_step = hw_df["step_ms"].mean()
                avg_render = hw_df["render_ms"].mean()
                max_vram = hw_df["torch_vram_alloc_mb"].max()

                hw_html = f"""
    <h2>5. Hardwarová Telemetrie & Profilace Výkonu (PC vs. Program)</h2>
    <div class="cards-grid">
        <div class="card" style="border-top-color: #00F0D2;">
            <div class="card-title">PROCESOR (CPU)</div>
            <div class="card-stat">PC: {avg_sys_cpu:.1f}% | App: {avg_proc_cpu:.1f}%</div>
            <div style="font-size: 11px; color: #91A2BC; margin-top: 4px;">Nízká zátěž procesoru, hlavní výpočty na GPU.</div>
        </div>
        <div class="card" style="border-top-color: #FFC319;">
            <div class="card-title">OPERAČNÍ PAMĚŤ (RAM)</div>
            <div class="card-stat">App: {avg_proc_ram:.0f} MB (PC: {avg_sys_ram:.1f} GB)</div>
            <div style="font-size: 11px; color: #91A2BC; margin-top: 4px;">Úsporný paměťový otisk v RAM.</div>
        </div>
        <div class="card" style="border-top-color: #FF2D73;">
            <div class="card-title">GRAFICKÁ KARTA (GPU)</div>
            <div class="card-stat">{avg_gpu:.0f}% Vytížení • {avg_temp:.0f}°C</div>
            <div style="font-size: 11px; color: #91A2BC; margin-top: 4px;">Tenzory AI: {max_vram:.1f} MB (masivní rezerva VRAM).</div>
        </div>
        <div class="card" style="border-top-color: #AF50FA;">
            <div class="card-title">PROFILACE LATENCE</div>
            <div class="card-stat">Krok: {avg_step:.1f}ms • UI: {avg_render:.1f}ms</div>
            <div style="font-size: 11px; color: #91A2BC; margin-top: 4px;">{'Stabilní běh do 60 FPS frame-budgetu' if avg_step <= 16.6 else 'Simulace je GPU compute bound'}.</div>
        </div>
    </div>
"""
            except Exception:
                pass

        html_content = f"""<!DOCTYPE html>
<html lang="cs">
<head>
    <meta charset="UTF-8">
    <title>Cyber-Tactics // AI Research Lab Report</title>
    <style>
        body {{
            background: #0B0E14;
            color: #EBF2FA;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Consolas, monospace;
            margin: 0;
            padding: 24px;
        }}
        h1, h2, h3 {{ font-family: Consolas, monospace; }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2C3C56;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0; color: #00F0D2; font-size: 24px; }}
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: #141A26;
            border: 1px solid #2C3C56;
            border-radius: 6px;
            padding: 16px;
        }}
        .stat {{ display: flex; justify-content: space-between; margin-bottom: 4px; font-size: 13px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #141A26;
            border: 1px solid #2C3C56;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 24px;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid #222C3D;
            font-size: 13px;
        }}
        th {{ background: #1B2333; color: #00F0D2; font-family: Consolas, monospace; }}
        .chart-box {{
            background: #141A26;
            border: 1px solid #2C3C56;
            border-radius: 6px;
            padding: 16px;
            text-align: center;
            margin-bottom: 24px;
        }}
        .chart-box img {{ max-width: 100%; height: auto; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⚡ CYBER-TACTICS // AI RESEARCH LAB REPORT</h1>
            <div style="color: #FFC319; font-size: 12px; margin-top: 4px;">Session: {self.session_id} • Vygenerováno: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</div>
        </div>
        <div style="text-align: right; color: #91A2BC; font-size: 13px;">
            Celkem epoch: <strong>{len(self.epochs_data)}</strong><br>
            Bojových událostí: <strong>{len(self.spatial_df) if self.spatial_df is not None else 0}</strong>
        </div>
    </div>

    <h2>1. Stav Frakcí a Vývoj Generálů</h2>
    <div class="cards-grid">
        {cards_html}
    </div>

    <h2>2. Vědecký Vizuální Dashboard (Křivky Učení, XAI & Heatmapy)</h2>
    <div class="chart-box">
        <img src="{img_name}" alt="AI Analysis Multi-Panel Report">
    </div>

    <h2>3. Ekonomika Armádního Rozpočtu // ROI Tříd</h2>
    <table>
        <thead>
            <tr>
                <th>Třída Jednotky</th>
                <th>Cena</th>
                <th>Nasazeno</th>
                <th>Prům. Fitness</th>
                <th>ROI / 1 b</th>
                <th>Celkový Damage</th>
                <th>Vyléčeno HP</th>
                <th>Míra Přežití</th>
            </tr>
        </thead>
        <tbody>
            {roi_rows}
        </tbody>
    </table>

    <h2>4. Poslední Odehrané Bitvy</h2>
    <table>
        <thead>
            <tr>
                <th>Epocha</th>
                <th>Scénář Bojiště</th>
                <th>Trvání</th>
                <th>Vítězná Frakce</th>
                <th>Důvod Ukončení</th>
            </tr>
        </thead>
        <tbody>
            {battles_rows}
        </tbody>
    </table>

    {hw_html}
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Cyber-Tactics Standalone AI Telemetry & Research Lab")
    parser.add_argument("--session", type=str, default="", help="Filtrovat session ID (např. 20261107_191129)")
    parser.add_argument("--gui", action="store_true", help="Otevřít interaktivní Matplotlib okno")
    parser.add_argument("--export", action="store_true", help="Vygenerovat CSV exporty")
    parser.add_argument("--dir", type=str, default="telemetry", help="Cesta ke složce telemetry")
    args = parser.parse_args()

    lab = TelemetryLab(session_filter=args.session, telemetry_dir=args.dir)

    # 1. Výpis do terminálu
    lab.print_terminal_summary()

    # 2. Generování PNG a HTML reportů
    if lab.epochs_data:
        png_path = lab.generate_visual_report()
        html_path = lab.generate_html_report()
        print(f"   [✓] Vizuální dashboard uložen: {png_path}")
        print(f"   [✓] Interaktivní HTML report:   {html_path}")

    # 3. Interaktivní okno pokud požadováno
    if args.gui and lab.epochs_data:
        # Přepnout na interaktivní backend a zobrazit
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        # Spustit vizuální zobrazení
        print("   Otevírám interaktivní analytické okno...")
        fig = plt.figure(figsize=(14, 8))
        # Jednoduché interaktivní zobrazení
        epochs_nums = [ep.get("epoch", idx + 1) for idx, ep in enumerate(lab.epochs_data)]
        for f in range(4):
            series = [ep.get("factions", [{}])[f].get("fitness_avg", 0.0) if f < len(ep.get("factions", [])) else 0.0 for ep in lab.epochs_data]
            plt.plot(epochs_nums, series, label=FACTION_NAMES[f], color=FACTION_COLORS[f], linewidth=2.0)
        plt.title("Vývoj Fitness Frakcí (Interaktivní náhled)")
        plt.xlabel("Epocha")
        plt.ylabel("Průměrná Fitness")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
