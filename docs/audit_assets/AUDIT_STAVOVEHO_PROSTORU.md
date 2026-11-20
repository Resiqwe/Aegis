# Architektonický audit stavového prostoru, normalizace a vektorové čistoty

> Statická analýza implementace v adresáři `test/src/` (bez úprav kódu).

---

## 1. Dimenze a bezrozměrnost stavového prostoru (Vstupy a výstupy sítě)

### 1.1 Vstupní senzorový vektor (`sensors` / `self.last_sensors`)
Senzorový tenzor má pevně danou dimenzi **`14`** prvků na jednotku, tj. tvar `(N, 14)` ([src/config.py:104-120](src/config.py#L104-L120), [src/engine_core.py:719-738](src/engine_core.py#L719-L738)).

| Index | Název (`SENSOR_NAMES`) | Vzorec a konstrukce v kódu | Rozsah & typ | Bezrozměrnost / anomálie |
| :---: | :--- | :--- | :--- | :--- |
| `0` | Cíl Vpřed/Vzad | `enemy_fwd = dir_enemy[:, 0] * cos_th + dir_enemy[:, 1] * sin_th` ([src/engine_core.py:677,723](src/engine_core.py#L677-L723)) | `[-1.0, 1.0]` spojité | **Bezrozměrné** (skalární projekce jednotkového vektoru na osu `heading`; `0.0` bez LoS) |
| `1` | Cíl Vpravo/Vlevo | `enemy_right = -dir_enemy[:, 0] * sin_th + dir_enemy[:, 1] * cos_th` ([src/engine_core.py:678,724](src/engine_core.py#L678-L724)) | `[-1.0, 1.0]` spojité | **Bezrozměrné** (ortogonální složka směru k cíli) |
| `2` | Blízkost cíle | `(1.0 - (dist_enemy / self.class_range)).clamp(-1.0, 1.0)` ([src/engine_core.py:725](src/engine_core.py#L725)) | `[-1.0, 1.0]` spojité | **Bezrozměrný poměr** (`1.0` = kontakt `0 px`, `0.0` = na hranici dostřelu; bez cíle `dist_enemy=9999.0` saturuje na `-1.0`) |
| `3` | Příchozí palba Vpřed/Vzad | `fire_fwd = incoming_fire_dir[:, 0] * cos_th + incoming_fire_dir[:, 1] * sin_th` ([src/engine_core.py:680,726](src/engine_core.py#L680-L726)) | `[-1.0, 1.0]` spojité | **Bezrozměrné** (egocentrická projekce jednotkového směru ke střelci s útlumem `mul_(0.85)` za tick, [src/engine_core.py:741,983-985](src/engine_core.py#L741-L985)) |
| `4` | Příchozí palba Vpravo/Vlevo | `fire_right = -incoming_fire_dir[:, 0] * sin_th + incoming_fire_dir[:, 1] * cos_th` ([src/engine_core.py:681,727](src/engine_core.py#L681-L727)) | `[-1.0, 1.0]` spojité | **Bezrozměrné** |
| `5` | Taktický povel Vpřed/Vzad | `order_fwd = dir_order[:, 0] * cos_th + dir_order[:, 1] * sin_th` ([src/engine_core.py:683,728](src/engine_core.py#L683-L728)) | `[-1.0, 1.0]` spojité | **Bezrozměrné** (jednotkový vektor k cílovému uzlu generála; u Medika s pacientem přepsán směrem k pacientovi, [src/engine_core.py:668-669](src/engine_core.py#L668-L669)) |
| `6` | Taktický povel Vpravo/Vlevo | `order_right = -dir_order[:, 0] * sin_th + dir_order[:, 1] * cos_th` ([src/engine_core.py:684,729](src/engine_core.py#L684-L729)) | `[-1.0, 1.0]` spojité | **Bezrozměrné** |
| `7` | Míra potlačení | `self.suppression` ([src/engine_core.py:530,730,1271](src/engine_core.py#L530-L1271)) | `[0.0, 1.0]` spojité | **Bezrozměrné, ale nesymetrické**: na rozdíl od většiny vstupů není mapováno do `[-1, 1]` |
| `8` | Zásobník munice | `(self.ammo_in_mag / self.class_mag_size) * 2.0 - 1.0` ([src/engine_core.py:731](src/engine_core.py#L731)) | Běžně `[-1.0, 1.0]` (až **`1.5`** s Overclockem) | **Bezrozměrný poměr**, ale s přesahem: `TECH_OVERCLOCK` násobí `ammo_in_mag *= 1.25` ([src/engine_core.py:294-295](src/engine_core.py#L294-L295)) bez úpravy jmenovatele `class_mag_size`, takže plný zásobník má hodnotu `1.5` |
| `9` | Zdraví (HP) | `(self.hp / self.class_max_hp) * 2.0 - 1.0` ([src/engine_core.py:732](src/engine_core.py#L732)) | `[-1.0, 1.0]` (u mrtvých `-1.0`) | **Bezrozměrný poměr** |
| `10` | Kryt Vpřed/Vzad | `cover_fwd = cover_dir[:, 0] * cos_th + cover_dir[:, 1] * sin_th` ([src/engine_core.py:699,733](src/engine_core.py#L699-L733)) | `[-1.0, 1.0]` spojité | **Bezrozměrné** (míří na těžiště nejbližšího obdélníku v `walls_gpu`, ignoruje hranu zdi i `pillars_gpu`, [src/engine_core.py:688-697](src/engine_core.py#L688-L697)) |
| `11` | Kryt Vpravo/Vlevo | `cover_right = -cover_dir[:, 0] * sin_th + cover_dir[:, 1] * cos_th` ([src/engine_core.py:700,734](src/engine_core.py#L700-L734)) | `[-1.0, 1.0]` spojité | **Bezrozměrné** |
| `12` | Koheze a Ping čety | `(ally_cnt / 8.0).clamp(0.0, 1.0) * 2.0 - 1.0` ([src/engine_core.py:705-707,735](src/engine_core.py#L705-L735)) | `[-1.0, 1.0]` kvantované | **Bezrozměrný poměr** počtu spojenců v pásmu `(1.0, 140.0) px`. **Rozpor s názvem:** `self.squad_ping` se do vstupu `[12]` ani jinam vůbec nepřenáší |
| `13` | Stav zóny dominance | `where(is_ours, 1.0, where(is_enemy, -1.0, 0.0))` ([src/engine_core.py:714-717,736](src/engine_core.py#L714-L736)) | `{-1.0, 0.0, 1.0}` diskrétní | **Bezrozměrný stav** vlastníka nejbližšího strategického uzlu |

- **Přítomnost surových fyzikálních jednotek:** **Žádné.** Síť pracuje v čistě egocentrické, bezrozměrné a translačně/rotačně invariantní soustavě. Surové souřadnice `pos` (`2400×1350 px`), rychlosti `vel`, surové `hp`, `class_range` či absolutní úhly `heading` do sítě přímo nevstupují.

---

### 1.2 Výstupní akční prostor (`ai_brains.py` a `engine_core.py`)
Výstupní tenzor `brain_out` má tvar **`(N, 7)`** a prochází smíšenými aktivacemi ([src/ai_brains.py:40-49](src/ai_brains.py#L40-L49)):

- **Indexy `0, 1, 2` (`Surge, Sway, Yaw`):** `torch.tanh(raw_out[:, :, 0:3])` -> `[-1, 1]`
- **Index `3` (`Trigger`):** `torch.sigmoid(raw_out[:, :, 3:4])` -> `[0, 1]`
- **Index `4` (`Stance`):** `torch.tanh(raw_out[:, :, 4:5])` -> `[-1, 1]`
- **Index `5` (`Ability`):** `torch.sigmoid(raw_out[:, :, 5:6])` -> `[0, 1]`
- **Index `6` (`Squad Ping`):** `torch.tanh(raw_out[:, :, 6:7])` -> `[-1, 1]`

#### Převod bezrozměrných výstupů na fyzikální veličiny:
1. **Lineární pohyb (`Surge`, `Sway`, `Stance`):**
   - Holonomická rotace tělesného vektoru do světa ([src/engine_core.py:801-803](src/engine_core.py#L801-L803)):
     `brain_mv_x = brain_surge * cos_th + brain_sway * (-sin_th)`
     `brain_mv_y = brain_surge * sin_th + brain_sway * cos_th`
   - Sloučení s odpuzováním od stěn a kohezí ([src/engine_core.py:806](src/engine_core.py#L806)):
     `move_raw = brain_world_vec * 1.35 + squad_force * 0.18 + wall_force * 1.60`
   - Prahované škálování podle postoje (`Stance`, [src/engine_core.py:879](src/engine_core.py#L879)):
     `stance_speed_mod = 1.30` (Sprint při `> 0.3`), `0.55` (Plížení při `< -0.3`), jinak `1.0`.
   - Celkový násobitel rychlosti ([src/engine_core.py:877-880](src/engine_core.py#L877-L880)):
     `speed_mult = 0.15` při staggeru, jinak `(1.0 - suppression * 0.40) * doc_speed_mod * stance_speed_mod`, kde `doc_speed_mod` je `1.12` (Rush) / `0.90` (Hold) / `1.0`.
   - Cílová rychlost v `px/tick` ([src/engine_core.py:882-885](src/engine_core.py#L882-L885)):
     `desired_vel = (effective_move_raw / norm.clamp(min=1.0)) * speed_mult * self.class_speed`
     kde `self.class_speed = [3.20, 3.40, 2.20, 2.50]` (`Grunt, Medic, Sniper, Juggernaut`, [src/engine_core.py:242](src/engine_core.py#L242)). Následná setrvačnost: `self.vel = lerp(self.vel, desired_vel, 0.26) * alive`.
2. **Úhlová rychlost otáčení (`Yaw`):**
   - `omega = aim_diff * 0.45 + brain_yaw * 0.20` pokud `has_combat_target`, jinak `brain_yaw * 0.45` (`rad/tick`) ([src/engine_core.py:894-895](src/engine_core.py#L894-L895)).
3. **Střelba a balistický impuls (`Trigger`):**
   - Prahová podmínka spouště: `brain_out[:, 3] > 0.58` ([src/engine_core.py:947](src/engine_core.py#L947)), vyžaduje zároveň `|aim_diff| < 45°`, `dist_enemy <= class_range`, nenulový zásobník a `cooldown == 0` ([src/engine_core.py:944-959](src/engine_core.py#L944-L959)).
   - Fyzikální konstanty střel dle třídy (`[Grunt, Medic, Sniper, Juggernaut]`):
     - Základní poškození `class_dmg`: `[28.0, 18.0, 48.0, 36.0]` ([src/engine_core.py:244](src/engine_core.py#L244))
     - Cooldown výstřelu `CONST_CLASS_COOLDOWNS`: `[8, 9, 22, 5]` ticků ([src/engine_core.py:132](src/engine_core.py#L132))
     - Kinetická energie střely `CLASS_KINETIC_ENERGY`: `[45.0, 22.0, 120.0, 95.0]` vs hmotnost cíle `CLASS_MASS`: `[1.0, 0.9, 0.85, 3.2]` ([src/engine/ballistics_system.py:40-41](src/engine/ballistics_system.py#L40-L41))
     - Výsledný impuls zpětného rázu/odhozu (`px/tick`): `impulse_mag = (effective_energy / target_mass) * 0.045` ([src/engine/ballistics_system.py:158](src/engine/ballistics_system.py#L158), zapsáno do `staging_buf.delta_vel` v [src/engine_core.py:1029](src/engine_core.py#L1029)).
4. **Taktická schopnost (`Ability`) a Signál (`Ping`):**
   - `Ability`: práh `brain_ability > 0.60` a `ability_cooldown <= 0.0` ([src/engine/combat_system.py:62](src/engine/combat_system.py#L62)) spouští třídně specifickou logiku (kouř `r=55 px / 6s`, plošné léčení `35 HP v 180 px`, nebo časovače buffů).
   - `Ping`: vyhlazuje se do `self.squad_ping = lerp(self.squad_ping, brain_ping * alive, 0.35)` ([src/engine_core.py:768](src/engine_core.py#L768)), ale dále v simulaci nikam nevstupuje.

---

## 2. Geometrie, poměrové škálování a šikmost rozdělení

### 2.1 Úhlová a projekční matematika ($\pi$, radiány, FOV, dopadové úhly)
Kód míchá dva odlišné přístupy – čistou projekci maticí/skalárním součinem a klasické goniometrické rozdíly v radiánech přes `atan2`:
- **Egocentrické senzory a pohyb** nepoužívají `atan2`, ale ortonormální projekci na vektor `(cos_th, sin_th)` a kolmici `(-sin_th, cos_th)` ([src/engine_core.py:674-700,801-802](src/engine_core.py#L674-L802)).
- **Detekce cíle (FOV) a míření** používají `atan2` s periodickým ovinutím do $[-\pi, \pi)$:
  `angle_diff = (atan2(dy, dx) - heading + pi) % (2*pi) - pi` ([src/engine_core.py:566-567,825-826,910-911,941-942](src/engine_core.py#L566-L942)).
  - Zorné pole (FOV): kužel `140°` (`|angle_diff| < radians(140) * 0.5`, tj. $\pm 70^\circ$), doplněný všesměrovým sluchem pro `dist_mat < 75.0 px` a omezený dosahem viditelnosti `CONST_CLASS_MAX_VIS = [380, 320, 680, 380] px` ([src/engine_core.py:130,568-576,912](src/engine_core.py#L130-L912)).
  - Střelecký kužel: `|aim_diff| < radians(45)` ([src/engine_core.py:828,944](src/engine_core.py#L828-L944)); zaměřovací laser Sniperů má `radians(50)` ([src/engine_core.py:1109](src/engine_core.py#L1109)).
- **Dopadové úhly v balistice** jsou řešeny čistě přes **skalární součin (dot product)** bez `atan2` ([src/engine/ballistics_system.py:102-109,184](src/engine/ballistics_system.py#L102-L184)):
  - `dot = dir_x * cos(target_heading) + dir_y * sin(target_heading)` (`-1.0` = čelní zásah, `+1.0` = zásah do zad).
  - `angle_factor = clamp((-dot + 1.0) * 0.5, min=0.15, max=1.0)`, takže čelní pancíř má 100 % tloušťky a týlový pancíř jen 15 % (`target_armor = armor * angle_factor`). Flankovací zásah se značí při `dot > 0.40`.
- **Line-of-Sight a kolize** jsou čistě analytické (Slab test AABB pro zdi + nejbližší bod úsečky `t = clamp((f·d)/|d|^2, 0, 1)` pro kruhové sloupy a kouř v [src/engine/spatial_system.py:33-94](src/engine/spatial_system.py#L33-L94)).

### 2.2 Nelineární vs. lineární / poměrové škálování
- **Vzdálenosti:**
  - Senzor vzdálenosti cíle `sensors[:, 2]` je **lineární poměr** `1 - d / class_range` ([src/engine_core.py:725](src/engine_core.py#L725)).
  - Kohezní síla čety `squad_force` používá **hyperbolickou váhu** $w_{ij} = 1 / \max(d_{ij}, 12.0)$ v okně `(12, 145) px`, ale výsledný posun dělí součtem vah (`squad_diff / weight_sum`), takže jde o vážený centroid okolí násobený `-0.35 * 0.18` ([src/engine_core.py:750-754,806](src/engine_core.py#L750-L806)).
  - Šance na zásah (`hit_prob`) **nemá žádnou spojitou křivku útlumu se vzdáleností** – používá pouze skokový práh `dist_enemy <= class_range` ([src/engine_core.py:958,988-994](src/engine_core.py#L958-L994)).
- **Průbojnost pancíře a poškození** ([src/engine/ballistics_system.py:128-138](src/engine/ballistics_system.py#L128-L138)):
  - Po částech definovaná funkce podle znaménka `delta_ap = bullet_ap - target_armor`:
    - Průraz (`delta_ap >= 0`): **lineární** `1.0 + clamp(delta_ap * 0.02, 0.0, 0.45)`.
    - Odolání pancíře (`delta_ap < 0`): **nelineární hyperbolický poměr** `clamp((bullet_ap / max(target_armor, 1.0)) * 0.60, 0.18, 0.90)`.
  - Zóny těla přidávají skokový multiplikátor `[2.50, 1.00, 0.55, 0.65]` (`Head, Torso, Arms, Legs`, [src/engine/ballistics_system.py:34,115-125](src/engine/ballistics_system.py#L34-L125)).
- **Fitness a odměny:** Všechny fitness přírůstky jsou čistě **lineární aditivní členy** (bez logaritmů či exponenciál):
  - Způsobené poškození: `+0.65 * dmg` ([src/engine_core.py:1073](src/engine_core.py#L1073))
  - Zásah obdržený mimo kryt: `-0.22 * dmg` ([src/engine_core.py:1056,1266](src/engine_core.py#L1056-L1266))
  - Zabití / Smrt / Přežití operace: `+30.0` / `-25.0` / `+18.0` ([src/engine_core.py:373,1285,1298](src/engine_core.py#L373-L1298))
  - Setrvání v krytu při potlačení `> 0.25`: `+0.15 / tick` ([src/engine_core.py:545-546](src/engine_core.py#L545-L546))
  - Léčení Medika: `+4.0 * actual_heal` pasivní ([src/engine_core.py:662](src/engine_core.py#L662)) a `+3.5 * total_healed` Bio-Pulz ([src/engine/combat_system.py:119](src/engine/combat_system.py#L119))
  - Anti-spin (když `|vel| < 0.20` bez cíle): `-|omega| * 0.18` ([src/engine_core.py:900-901](src/engine_core.py#L900-L901))
  - Sniper kempování (`> 8.0 s` stání): `-1.8 * dt` ([src/engine_core.py:1128-1130](src/engine_core.py#L1128-L1130))
  - Obsazování uzlů (každý 6. tick): `+0.45*6*bonus` / `+0.65*6*bonus` / `+0.25*6`, kde Pěšák má `bonus = 2.5` ([src/engine_core.py:1170-1197](src/engine_core.py#L1170-L1197)).
  - **Strategický výběr generála (`FactionGeneralBrain`):** používá poměrové ROI `avg_c_fit / CLASS_COSTS[c]` ([src/ai_brains.py:337](src/ai_brains.py#L337)), EMA vyhlazování (`alpha = 0.20`), **z-score normalizaci** `(ema - mean) / (std + 1e-5)` ([src/ai_brains.py:150-159](src/ai_brains.py#L150-L159)) a následný **softmax** pro pravděpodobnosti nákupu tříd ([src/ai_brains.py:146](src/ai_brains.py#L146)).

### 2.3 Statistická rozdělení a asymetrie v evoluci a mutacích (`ai_brains.py`)
- **Inicializace vah (`_init_all_banks`, [src/ai_brains.py:206-247](src/ai_brains.py#L206-L247)):**
  - Základní rozdělení je **symetrický Gaussův šum** s nulovým středem a zmenšující se směrodatnou odchylkou po vrstvách: `W1 ~ N(0, 0.28²)`, `W2 ~ N(0, 0.25²)`, `W3 ~ N(0, 0.22²)`; biasy jsou nulové (`b1=b2=b3=0`).
  - Následuje ale silné **deterministické přepsání konkrétních prvků** (vrozené reflexy v rozsahu `-0.90` až `+1.40` v `W1`, `+1.05` na diagonále `W2`, a konkrétní akční spoje v `W3`).
- **Selekce a křížení (`evolve_epoch`, [src/ai_brains.py:374-412](src/ai_brains.py#L374-L412)):**
  - **Elitismus:** horních `n_elite = max(2, min(5, c_count // 3))` jedinců dané třídy se klonuje beze změny do prvních slotů třídy (`base_slot : base_slot + n_elite`).
  - **Rodičovská selekce (truncation + uniformní losování):** `parent_pool` tvoří horních `max(n_elite + 1, c_count // 2)` jedinců; výběr dvojic rodičů uvnitř této horní poloviny je **čistě uniformní** (`torch.randint(0, n_parents, ...)`, bez vážení podle velikosti fitness).
  - **Křížení:** pro váhy symetrický **uniformní crossover 50/50** (`torch.rand(...) > 0.5`), pro biasy **aritmetický průměr** `0.5 * (b_p1 + b_p2)` (což tlumí rozptyl biasů proti rozptylu vah).
- **Mutace a L2 útlum ([src/ai_brains.py:414-433](src/ai_brains.py#L414-L433)):**
  - Výchozí `mutation_rate = 0.08` ([src/ai_brains.py:318](src/ai_brains.py#L318)) je **asymetrická dle výsledku bitvy**: vítězná frakce má `m = 0.85 * 0.08 = 0.068`, poražené frakce `m = 1.20 * 0.08 = 0.096`.
  - Při stagnaci (`cur_best <= prev_best * 1.015` po `>= 3` epochy, [src/ai_brains.py:346,355](src/ai_brains.py#L346-L355)) se `m` násobí `temp_mult = min(3.2, 1.8 + 0.3 * stag_epochs)` a na **15 % vah** (`torch.rand_like < 0.15`) proběhne skokový reset z `N(0, 0.40²)` v `W1` a `N(0, 0.35²)` v `W2` (`W3` šokový reset nemá).
  - Běžná mutace je aditivní **symetrický Gaussův šum** (`randn_like * m` na váhy, `*(0.5*m)` na biasy), následovaný jednotným **L2 weight decay `mul_(0.988)`** na všech maticích `W1, W2, W3` dětí (elity decay nemají).

---

## 3. Audit „epicyklů“ a kapacita neuronových sítí

### 3.1 Tvrdě zadrátované výjimky, speciální masky a mrtvé proměnné

1. **Klasové „epicykly“ v balistice ([src/engine/ballistics_system.py:151-155,172](src/engine/ballistics_system.py#L151-L172)):**
   - I když existuje parametrická tabulka `CLASS_STAGGER_THRESHOLDS = [40.0, 35.0, 35.0, 95.0]`, knockback natvrdo větví maskou `(t_cls == 3) & (kin_energy < 95.0)`, která Juggernautovi vynuluje veškerou kinetickou energii místo parametrického tlumení hmotností/prahem.
   - Potlačení má přímo v rovnici natvrdo třídní podmínku `torch.where(t_cls == 3, 0.16, 0.35)` místo pole atributů třídy.
2. **Skriptový override rozkazu a léčení Medika ([src/engine_core.py:619-669](src/engine_core.py#L619-L669)):**
   - Pouze pro `CLASS_MEDIC` běží separátní vyhledávání zraněných do `450 px`, automatické pasivní léčení (`42.0 * dt` do `260 px` bez účasti mozku) a **přepis `dir_order`** směrem k pacientovi ([src/engine_core.py:669](src/engine_core.py#L669)).
3. **Sniper-specifická pravidla ([src/engine_core.py:989,1096-1130](src/engine_core.py#L989-L1130)):**
   - Sniper má nižší postih přesnosti proti cíli v krytu (`0.60` vs `0.48` u všech ostatních tříd) a **jako jediná třída** podléhá penalizaci za stání na místě (`sniper_camp_timer > 8.0 s -> fitness -= 1.8 * dt`).
4. **Grunt / Juggernaut bonusy v cílech a percích:**
   - Pěšák (`CLASS_GRUNT`) dostává při obsazování uzlů hardcoded násobitel `2.5x` fitness ([src/engine_core.py:1168-1170](src/engine_core.py#L1168-L1170)).
   - Juggernaut dává `3.0x` poškození zničitelným branám ([src/engine/objectives_system.py:97](src/engine/objectives_system.py#L97)) a jako jediná třída má na Levelu 9 generála dedikovaný perk `+15 % HP` ([src/engine_core.py:257-259](src/engine_core.py#L257-L259)).
5. **Skrytá nekonzistence při `restart_all=True` ([src/engine_core.py:202](src/engine_core.py#L202) vs [src/ai_brains.py:66](src/ai_brains.py#L66)):**
   - Zatímco `FactionGeneralBrain.__init__` začíná s vyváženými váhami `[1.0, 0.7, 0.6, 0.5]`, metoda `start_new_operation(restart_all=True)` je přepíše na **`[1.0, -10.0, -10.0, -10.0]`**, čímž tvrdě potlačí nákup Mediků, Sniperů a Juggernautů po resetu simulace.
6. **Sekvenční Python `if/elif` v schopnostech + nefunkční („mrtvé“) buffy a technologie:**
   - `CombatSystem.process_abilities` ([src/engine/combat_system.py:69-132](src/engine/combat_system.py#L69-L132)) prochází aktivace v Python cyklu s `.item()` synchronizacemi a 4 větvemi `if/elif`.
   - **Kritický nález:** Ačkoliv `combat_system.py` nastavuje a odpočítává **`sniper_focus_timer` (`5.0 s`)** a **`fortress_shield_timer` (`6.0 s`)** ([src/engine/combat_system.py:53-54,124,130](src/engine/combat_system.py#L53-L130)), žádný jiný modul v `engine_core.py` ani `ballistics_system.py` tyto dva tenzory nečte – obě schopnosti dávají jen jednorázový fitness bonus (`+2.0` / `+2.5`), ale žádný bojový efekt.
   - Obdobně technologie **`TECH_RADAR`** ([src/config.py:73](src/config.py#L73)) se při boji nikde neaplikuje na dosah viditelnosti ani LoS.
7. **Fyzikální a stavové „berličky“ mimo neuronovou síť v `_step_internal`:**
   - **Částečný auto-aim při LoS:** `omega = aim_diff * 0.45 + brain_yaw * 0.20` ([src/engine_core.py:894](src/engine_core.py#L894)) aktivně natáčí jednotku k nepříteli nezávisle na `brain_yaw`.
   - **Deterministický automat přebíjení:** Přebíjení při prázdném zásobníku i automatické taktické přebití v krytu pod `45 %` kapacity ([src/engine_core.py:925-928](src/engine_core.py#L925-L928)).
   - **Vektory sil prostředí v pohybu:** `move_raw` přimíchává k tahu sítě odpuzování stěn (`wall_force * 1.60`) a kohezi (`squad_force * 0.18`) ([src/engine_core.py:806](src/engine_core.py#L806)).
   - **Telemetrický ArgMax selektor stavů (`utility` -> `self.unit_state`):** Počítá se každý krok přes 6 třídních/situačních užitkových funkcí (`IDLE`, `ENGAGE`, `COVER_RETREAT`, `SNIPER_LOCK`, `TRIAGE_HEAL`, `FORTRESS_PUSH`, [src/engine_core.py:817-870](src/engine_core.py#L817-L870)), ale jak potvrzuje komentář na řádku `869`, slouží už jen pro HUD telemetrii a nezasahuje do pohybu.

---

### 3.2 Přesná šířka, hloubka a kapacita neuronové sítě (`ai_brains.py`)
Síť je **3-vrstvý MLP (2 skryté vrstvy)** s architekturou `14 -> 64 -> 64 -> 7` ([src/ai_brains.py:37-48,178-193](src/ai_brains.py#L37-L193)), alokovaný pro `MAX_UNITS = 800` slotů (`4 frakce × 4 třídy × 50 slotů`):

- **Vrstva 1 (Percepce, aktivace `tanh`):** `W1: (800, 14, 64)`, `b1: (800, 1, 64)` -> $14 \times 64 + 64 = \mathbf{960}$ parametrů (`896` vah + `64` biasů)
- **Vrstva 2 (Taktická syntéza, aktivace `tanh`):** `W2: (800, 64, 64)`, `b2: (800, 1, 64)` -> $64 \times 64 + 64 = \mathbf{4\,160}$ parametrů (`4\,096` vah + `64` biasů)
- **Vrstva 3 (Akce, smíšeně `tanh` / `sigmoid`):** `W3: (800, 64, 7)`, `b3: (800, 1, 7)` -> $64 \times 7 + 7 = \mathbf{455}$ parametrů (`448` vah + `7` biasů)

**Porovnání kapacity vůči I/O:**
- **Celkový počet trénovatelných parametrů na 1 mozek:** **`5 575`** (`5 440` vah + `135` biasů); celá banka ve VRAM nese `4 460 000` parametrů (`~17,0 MB` ve `float32`).
- Vzhledem ke **`14` vstupům** a **`7` výstupům** (`21` rozhraní) jde o poměr **~398 parametrů na 1 vstup** (resp. **~265 parametrů na součet I+O**), což je při čistě bezgradientové neuroevoluci s populací nejvýše `50` jedinců na třídu a frakci značně předimenzovaný prostor – to vysvětluje nutnost silných vrozených reflexů v `_init_all_banks`, auto-aim asistence v `omega` i deterministických automatů kolem sítě.
