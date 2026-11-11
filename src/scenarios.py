"""
scenarios.py - Taktická Bojiště, Terény a Herní Režimy (1 GB VRAM & Multicore Edition)
Obsahuje:
  1. SCENARIO_DOMINATION - Boj o 5 centrálních energetických nodů (Control Nodes)
  2. SCENARIO_CITADEL - Asymetrická obrana těžce opevněné centrální citadely
  3. SCENARIO_CHASM - Zákopová válka přes kaňon se 3 strategickými mosty
Každý scénář definuje:
  - Polygonové / obdélníkové zdi a bunkry (x1, y1, x2, y2)
  - Energetické uzly (uzel_x, uzel_y, radius, reward_type)
  - Spawnovací zóny frakcí (včetně orientace a ochranných zón)
  - CPU Flow-Field mřížku pro plynulé obcházení překážek bez zasekávání u zdí
"""

import numpy as np
import torch

SCENARIO_DOMINATION = 0
SCENARIO_CITADEL = 1
SCENARIO_CHASM = 2

SCENARIO_NAMES = [
    "RECEPTOR DOMINATION (5 Energetických Nodů)",
    "CITADEL SIEGE (Asymetrická Pevnost)",
    "THE CHASM // FRONTLINE (Boj o Mosty)",
]


class MapScenario:
    def __init__(self, scenario_type: int, width: int = 2400, height: int = 1350):
        self.type = scenario_type
        self.W = width
        self.H = height
        self.name = SCENARIO_NAMES[scenario_type]

        # Stěny a překážky jako polygonové obdélníky [x1, y1, x2, y2]
        self.walls = []
        # Kruhové sloupy / reaktory [cx, cy, r]
        self.pillars = []
        # Strategické kontrolní body [cx, cy, r, capture_progress, owner_team]
        # owner_team: -1 = neutrální, 0..3 = frakce
        self.nodes = []
        # Základny [faction_id, cx, cy, radius]
        self.bases = []
        # Zničitelné brány pevnosti: {"rect": [x1, y1, x2, y2], "hp", "team", "name"}
        self.gates = []
        self.gate_wall_indices = []

        self._build_scenario()

    def _build_scenario(self):
        W, H = self.W, self.H

        if self.type == SCENARIO_DOMINATION:
            # 1. Čtyři rohové základny
            self.bases = [
                {"team": 0, "x": W * 0.12, "y": H * 0.16, "r": 90.0},
                {"team": 1, "x": W * 0.88, "y": H * 0.84, "r": 90.0},
                {"team": 2, "x": W * 0.88, "y": H * 0.16, "r": 90.0},
                {"team": 3, "x": W * 0.12, "y": H * 0.84, "r": 90.0},
            ]

            # 2. 5 Strategických Energetických Nodů (Střed + 4 křídla)
            self.nodes = [
                {"id": 0, "x": W * 0.50, "y": H * 0.50, "r": 65.0, "owner": -1, "cap": 0.0, "name": "NEXUS PRIME"},
                {"id": 1, "x": W * 0.50, "y": H * 0.22, "r": 48.0, "owner": -1, "cap": 0.0, "name": "NODE SEVERNÍ"},
                {"id": 2, "x": W * 0.50, "y": H * 0.78, "r": 48.0, "owner": -1, "cap": 0.0, "name": "NODE JIŽNÍ"},
                {"id": 3, "x": W * 0.28, "y": H * 0.50, "r": 48.0, "owner": -1, "cap": 0.0, "name": "NODE ZÁPADNÍ"},
                {"id": 4, "x": W * 0.72, "y": H * 0.50, "r": 48.0, "owner": -1, "cap": 0.0, "name": "NODE VÝCHODNÍ"},
            ]

            # 3. Polygonové taktické zdi a kryty (tvoří chokepointy a uličky)
            self.walls = [
                # Středový kříž s průchody
                [W * 0.38, H * 0.34, W * 0.44, H * 0.37],
                [W * 0.56, H * 0.34, W * 0.62, H * 0.37],
                [W * 0.38, H * 0.63, W * 0.44, H * 0.66],
                [W * 0.56, H * 0.63, W * 0.62, H * 0.66],
                # Rohové barikády kryjící přístupy
                [W * 0.25, H * 0.26, W * 0.27, H * 0.38],
                [W * 0.73, H * 0.26, W * 0.75, H * 0.38],
                [W * 0.25, H * 0.62, W * 0.27, H * 0.74],
                [W * 0.73, H * 0.62, W * 0.75, H * 0.74],
            ]

            # 4. Reaktory a masivní pilíře
            self.pillars = [
                [W * 0.34, H * 0.22, 38.0],
                [W * 0.66, H * 0.22, 38.0],
                [W * 0.34, H * 0.78, 38.0],
                [W * 0.66, H * 0.78, 38.0],
            ]

        elif self.type == SCENARIO_CITADEL:
            # Centrální Pevnost (Obránce: Tým 0 uprostřed, Útočníci: Týmy 1, 2, 3 v rozích)
            self.bases = [
                {"team": 0, "x": W * 0.50, "y": H * 0.50, "r": 110.0},
                {"team": 1, "x": W * 0.12, "y": H * 0.16, "r": 85.0},
                {"team": 2, "x": W * 0.88, "y": H * 0.16, "r": 85.0},
                {"team": 3, "x": W * 0.50, "y": H * 0.88, "r": 85.0},
            ]

            self.nodes = [
                {"id": 0, "x": W * 0.50, "y": H * 0.50, "r": 60.0, "owner": 0, "cap": 1.0, "name": "CITADEL CORE"},
                {"id": 1, "x": W * 0.30, "y": H * 0.32, "r": 42.0, "owner": -1, "cap": 0.0, "name": "PŘEDNÍ BRÁNA Z"},
                {"id": 2, "x": W * 0.70, "y": H * 0.32, "r": 42.0, "owner": -1, "cap": 0.0, "name": "PŘEDNÍ BRÁNA V"},
            ]

            # Masivní hradby Citadely se 4 zničitelnými branami
            self.walls = [
                # Severní hradba (mezera = severní brána)
                [W * 0.36, H * 0.36, W * 0.46, H * 0.39],
                [W * 0.54, H * 0.36, W * 0.64, H * 0.39],
                # Jižní hradba (mezera = jižní brána)
                [W * 0.36, H * 0.61, W * 0.46, H * 0.64],
                [W * 0.54, H * 0.61, W * 0.64, H * 0.64],
                # Západní křídlo rozdělené bránou
                [W * 0.34, H * 0.39, W * 0.37, H * 0.46],
                [W * 0.34, H * 0.54, W * 0.37, H * 0.61],
                # Východní křídlo rozdělené bránou
                [W * 0.63, H * 0.39, W * 0.66, H * 0.46],
                [W * 0.63, H * 0.54, W * 0.66, H * 0.61],
                # Vnější překážky pro útočníky
                [W * 0.18, H * 0.48, W * 0.24, H * 0.52],
                [W * 0.76, H * 0.48, W * 0.82, H * 0.52],
            ]

            self.gates = [
                {"rect": [W * 0.46, H * 0.352, W * 0.54, H * 0.398], "hp": 450.0, "team": 0, "name": "SEVERNÍ BRÁNA"},
                {"rect": [W * 0.46, H * 0.602, W * 0.54, H * 0.648], "hp": 450.0, "team": 0, "name": "JIŽNÍ BRÁNA"},
                {"rect": [W * 0.34, H * 0.46, W * 0.37, H * 0.54], "hp": 450.0, "team": 0, "name": "ZÁPADNÍ BRÁNA"},
                {"rect": [W * 0.63, H * 0.46, W * 0.66, H * 0.54], "hp": 450.0, "team": 0, "name": "VÝCHODNÍ BRÁNA"},
            ]
            for gate in self.gates:
                gate["max_hp"] = gate["hp"]
                gate["destroyed"] = False
                self.gate_wall_indices.append(len(self.walls))
                self.walls.append(list(gate["rect"]))

            self.pillars = [
                [W * 0.35, H * 0.37, 28.0],
                [W * 0.65, H * 0.37, 28.0],
                [W * 0.35, H * 0.63, 28.0],
                [W * 0.65, H * 0.63, 28.0],
            ]

        else:  # SCENARIO_CHASM
            # Propast/Kaňon dělí mapu na sever (Týmy 0, 2) a jih (Týmy 1, 3)
            self.bases = [
                {"team": 0, "x": W * 0.16, "y": H * 0.18, "r": 85.0},
                {"team": 2, "x": W * 0.84, "y": H * 0.18, "r": 85.0},
                {"team": 1, "x": W * 0.16, "y": H * 0.82, "r": 85.0},
                {"team": 3, "x": W * 0.84, "y": H * 0.82, "r": 85.0},
            ]

            self.nodes = [
                {"id": 0, "x": W * 0.50, "y": H * 0.50, "r": 55.0, "owner": -1, "cap": 0.0, "name": "CENTRÁLNÍ MOST"},
                {"id": 1, "x": W * 0.22, "y": H * 0.50, "r": 45.0, "owner": -1, "cap": 0.0, "name": "ZÁPADNÍ PŘECHOD"},
                {"id": 2, "x": W * 0.78, "y": H * 0.50, "r": 45.0, "owner": -1, "cap": 0.0, "name": "VÝCHODNÍ PŘECHOD"},
            ]

            # Masivní skalní stěny tvořící kaňon (propustné pouze přes 3 mosty)
            self.walls = [
                # Kaňon mezi Z a Středním mostem
                [W * 0.28, H * 0.46, W * 0.44, H * 0.54],
                # Kaňon mezi Středním a V mostem
                [W * 0.56, H * 0.46, W * 0.72, H * 0.54],
                # Vnější zátarasy
                [W * 0.04, H * 0.46, W * 0.16, H * 0.54],
                [W * 0.84, H * 0.46, W * 0.96, H * 0.54],
            ]

            self.pillars = [
                [W * 0.28, H * 0.32, 34.0],
                [W * 0.72, H * 0.32, 34.0],
                [W * 0.28, H * 0.68, 34.0],
                [W * 0.72, H * 0.68, 34.0],
            ]

    def build_gpu_tensors(self, device: torch.device):
        """Vytvoří GPU tenzory stěn, pilířů a nodů pro raymarching a kolize."""
        # Walls: (Nw, 4) -> [x1, y1, x2, y2]
        walls_t = torch.tensor(self.walls, dtype=torch.float32, device=device) if self.walls else torch.zeros((0, 4), device=device)
        # Pillars: (Np, 3) -> [cx, cy, r]
        pillars_t = torch.tensor(self.pillars, dtype=torch.float32, device=device) if self.pillars else torch.zeros((0, 3), device=device)

        node_data = []
        for n in self.nodes:
            node_data.append([n["x"], n["y"], n["r"], float(n["owner"]), n["cap"]])
        nodes_t = torch.tensor(node_data, dtype=torch.float32, device=device) if node_data else torch.zeros((0, 5), device=device)

        return walls_t, pillars_t, nodes_t


def compute_cpu_flow_field(W: int, H: int, walls: list, pillars: list, cell_size: int = 32) -> np.ndarray:
    """
    Spouští se na CPU (Multithreading): Spočítá vektorové odpudivé pole (Repulsion Field)
    od všech stěn, pilířů a okrajů mapy.
    Výsledek je pole (GH, GW, 2), kde buňky dávají přirozený vektor směru, kterým
    má jednotka uhnout od překážky nebo okraje mapy!
    """
    gw = (W + cell_size - 1) // cell_size
    gh = (H + cell_size - 1) // cell_size

    field = np.zeros((gh, gw, 2), dtype=np.float32)

    # Středy buněk
    xs = np.arange(gw) * cell_size + cell_size * 0.5
    ys = np.arange(gh) * cell_size + cell_size * 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)

    # 1. Měkké odpuzování od okrajů bojiště (Konec lepení na zdi!)
    edge_margin = 85.0
    field[:, :, 0] += np.clip((edge_margin - grid_x) / edge_margin, 0.0, 1.0) * 1.6
    field[:, :, 0] -= np.clip((grid_x - (W - edge_margin)) / edge_margin, 0.0, 1.0) * 1.6
    field[:, :, 1] += np.clip((edge_margin - grid_y) / edge_margin, 0.0, 1.0) * 1.6
    field[:, :, 1] -= np.clip((grid_y - (H - edge_margin)) / edge_margin, 0.0, 1.0) * 1.6

    # 2. Odpuzování od obdélníkových stěn
    for w in walls:
        x1, y1, x2, y2 = w
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        hw = (x2 - x1) * 0.5 + 40.0
        hh = (y2 - y1) * 0.5 + 40.0

        dx = grid_x - cx
        dy = grid_y - cy
        inside_x = np.abs(dx) < hw
        inside_y = np.abs(dy) < hh
        inside = inside_x & inside_y
        if np.any(inside):
            sign_x = np.sign(dx)
            sign_y = np.sign(dy)
            force_x = (hw - np.abs(dx)) / hw
            force_y = (hh - np.abs(dy)) / hh
            field[inside, 0] += sign_x[inside] * force_x[inside] * 2.2
            field[inside, 1] += sign_y[inside] * force_y[inside] * 2.2

    # 3. Odpuzování od kruhových pilířů
    for p in pillars:
        px, py, pr = p
        eff_r = pr + 45.0
        dist = np.hypot(grid_x - px, grid_y - py)
        inside = dist < eff_r
        if np.any(inside):
            d_safe = np.maximum(dist, 1.0)
            push = (eff_r - dist) / eff_r
            field[inside, 0] += ((grid_x[inside] - px) / d_safe[inside]) * push[inside] * 2.4
            field[inside, 1] += ((grid_y[inside] - py) / d_safe[inside]) * push[inside] * 2.4

    # Normalizace
    mags = np.hypot(field[:, :, 0], field[:, :, 1])
    overshoot = mags > 1.0
    field[overshoot, 0] /= mags[overshoot]
    field[overshoot, 1] /= mags[overshoot]

    return field
