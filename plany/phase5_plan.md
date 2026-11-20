Analyzed engine_core.py:L535-690

Provedl jsem cílenou analýzu přesně podle rozsahu řádků z `xray` a `profile` (bez jakékoliv úpravy produkčního kódu). Tady je přesná diagnóza všech 3 bottlenecků + návrh **Fáze 5**:

---

## 1. Diagnóza příčin (kde mizí čas a 1 126 alokací/tik)

### A. `combat.process_abilities` (**3,76 ms** — [src/engine/combat_system.py](file:///C:/Users/resiq/Documents/projekty/pokec/test/src/engine/combat_system.py#L62-L140))
* **Python smyčka přes mediky (`for m_uid in med_indices:`, L85–102):** Pro každého medika spouštějícího Bio-Pulz se zvlášť volá `torch.where(team_mask)`, `torch.norm`, `torch.full_like` a `index_add_` → místo 1 maticové operace `(M, N)` vznikají desítky malých GPU kernelů a alokací.
* **Python smyčka přes grunty se 7× `.item()` (L109–139):** Každý grunt spouštějící kouřovnici dělá 7 synchronizačních `.item()` (`g_uid`, `teams`, `team_smokes`, `pos[0/1]`, 2× `torch.rand(1).item()`), což zastavuje GPU pipeline.
* **Zbytečné alokace skalárů:** `torch.full_like(ability_cooldown, 20.0)` / `22.0` / `15.0` alokuje zbytečně celé `(N,)` tenzory, ačkoliv `torch.where(mask, 20.0, ability_cooldown)` bere skalár přímo.

### B. `sensors_and_ego` (**~4,74 ms** bez mozku — [src/engine_core.py](file:///C:/Users/resiq/Documents/projekty/pokec/test/src/engine_core.py#L536-L785))
* **Největší plýtvání – `O(K²)` výpočty `angle_to_tar`, `in_fov`, `can_spot`, `vis_dists` běží KAŽDÝ tik (L564–582), i když se `vis_dists` použije jen každý 3. tik (`if need_full_los:`)!** Při `K = 800` se tak 2 ze 3 tiků zbytečně alokuje a počítá 5 matic `800×800` (~15 MB dočasných tenzorů/tik), které se hned zahodí.
* **Skryté CPU–GPU synchronizace (`.any()` a `torch.where(mask)[0]`):** `L577` (`f_in_alive.any()`), `L630` (`has_cached_tar.any()`), `L660–681` (4× `.any()` u medika).
* **Opakované alokace každý tik:** `dir_enemy`, `dist_enemy`, `target_enemy`, `medic_patient_dir`, `medic_has_patient` se alokují znovu každý tik místo předalokovaných bufferů; podobně 6× `torch.zeros_like` / `torch.full_like` pro skaláry.

### C. `targeting_and_weapons` (**~4,71 ms** bez kolizí/balistiky — [src/engine_core.py](file:///C:/Users/resiq/Documents/projekty/pokec/test/src/engine_core.py#L865-L1085))
* **Zbytečné `torch.where(mask)[0]` a `.any()` synchronizace:**
  * Kontrola ztráty FOV (`L953–966`): `if has_tar.any():` + `torch.where(has_tar)[0]` + `if lost_fov.any():` → přitom jde napsat čistě vektorově bez jediného `.any()` a `nonzero` kernelu!
  * Kontrola míření (`L983–992`): `valid_tar_idx = torch.where(alive & valid_tar)[0]` → zbytečný `nonzero` sync, stačí `tar_safe = tar_for_shoot.clamp(0, N - 1)` a přímý výpočet `on_target` na všech `N` jednotkách najednou.
  * Štíty (`L1068`): `if b_res["shield_absorbed"].any():` zbytečně synchronizuje GPU.
* **Zbytečné alokace:** 6× `torch.zeros_like(self.suppression)` ve stavovém registru (`L878–914`) místo skaláru `0.0` + 5× `torch.zeros((0, 2))` každý tik (`L1010–1014`) místo sdílené prázdné konstanty `_empty_xy`.

---

## 2. Návrh implementačního plánu – Fáze 5 (GPU Optimalizace & Snížení alokací)

Abychom udrželi **100% shodu v `bench --compare`** (`EXACT` / `WITHIN_TOL`), navrhuji 3 čisté, nezávislé kroky (po každém pustíme `bench --compare` a `profile --compact`):

### Krok 5.1: Vektorizace `combat.process_abilities` (`src/engine/combat_system.py`)
* **Medic Bio-Pulz:** Nahradit `for m_uid in med_indices:` jedním maticovým výpočtem `torch.cdist(pos[med_indices], pos)` tvaru `(M, N)` → 1× `index_add_` do `staging_buf.delta_hp` a 1× přičtení `healing_done` / `fitness`.
* **Grunt Kouř:** Místo per-grunt `.item()` stáhnout jen aktivující grunty (`grunt_indices`) jedním dávkovým `.detach().cpu().numpy()` (včetně pozic a týmů) a v `visual_mode` použít `np.random` místo `torch.rand(1).item()` (tím zmizí všech **7× `.item()`** z `combat_system.py`!).
* Nahradit `torch.full_like(..., const)` za skaláry v `torch.where(...)`.
* *Očekávaný zisk:* `combat.process_abilities` z **3,76 ms → < 0,4 ms** (**~−3,3 ms/tik**).

### Krok 5.2: Eliminace zbytečných `O(K²)` výpočtů v `sensors_and_ego` (`src/engine_core.py`, L536–785)
* **Přesunout `angle_to_tar`, `angle_diff`, `in_fov`, `can_spot`, `vis_dists` dovnitř `if need_full_los:`** — 2 ze 3 tiků se vůbec nepočítají (nulová změna výsledku, protože mimo `need_full_los` se tyhle tenzory vůbec nečtou!).
* Nahradit `torch.full_like` / `torch.zeros_like` za skaláry a předalokovat pracovní buffery (`_buf_dir_enemy`, `_buf_dist_enemy`, `_buf_target_enemy`).
* *Očekávaný zisk:* `sensors_and_ego` z **5,01 ms → ~2,0–2,5 ms** (**~−2,5 ms/tik**) + velký pokles `alloc_calls_per_tick`.

### Krok 5.3: Bezvětvené `targeting_and_weapons` (`src/engine_core.py`, L865–1075)
* Přepsat `lost_fov` (L952–966) a `on_target` (L982–992) bez `torch.where(mask)[0]` a bez `.any()` pomocí `tar_safe = target_enemy.clamp(0, N - 1)`.
* Nahradit 6× `torch.zeros_like(self.suppression)` ve stavovém registru za `0.0` a znovupoužít prázdné tenzory `_empty_xy` / `_empty_i8` pro vizuální stopy.
* *Očekávaný zisk:* `targeting_and_weapons` z **7,50 ms → ~4,5–5,0 ms** (**~−2,5 ms/tik**).

---

### Celkový odhad po Fázi 5
* **Čas tiku:** z `~16,3 ms` (`61 TPS`) dolů k **`~7–8 ms` (`120–140 TPS`, tj. 2× zrychlení)** při zachování deterministické shody v `aegis_tool.py bench --compare` (`EXACT` / `WITHIN_TOL`) a `aegis_tool.py health` (`exit=0`).
* *(Další případná Fáze 6 pak může vzít `spatial.resolve_obstacle_collisions` – 1,92 ms – a odstranit zbývající 2× `.item()`.)*

Chceš, abych rovnou přešel k implementaci **Fáze 5** (krok po kroku s ověřením přes `bench --compare` a `profile --compact`), nebo si chceš nejdřív ještě něco doladit v návrhu?
