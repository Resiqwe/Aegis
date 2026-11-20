# Implementační plán – Fáze 2: Eliminace skrytých CPU–GPU synchronizací v horké smyčce

## 1. Shrnutí problému a ověřená diagnóza
* **Dotčený soubor:** `src/engine_core.py` (řádky 504–520, 1200–1266, 1340–1375)
* **Skutečné příčiny GPU stallů (`cudaStreamSynchronize`) v simulačním tiku:**
  1. **Re-entrantní zamykání v `step()` → `_step_internal()` (řádky 504–509) a `_update_gates()` (řádek 1256):** Každý simulační tik opakovaně vstupuje do `self._engine_lock` (`RLock`), ačkoli interní metody jsou volány výhradně pod již získaným zámkem.
  2. **Python smyčka přes uzly s `.item()` a `.any()` (řádky 1203–1251):** Pro každý uzel na mapě (`for i in range(num_nodes):`) kód volá `int(self.nodes_gpu[i, 3].item())` a opakovaně větví přes `if in_node.any():` a `if f_in.any():`. Každé `.item()` a `.any()` zastaví CPU vlákno a vynutí vyprázdnění fronty CUDA kernelů uprostřed tiku.
  3. **Smyčka přes zabité jednotky s opakovaným `.item()` (řádky 1344–1364):** Při každém zabití jednotky v tiku kód v cyklu volá opakovaně `.item()` (`victim_idx`, `last_attacker_id`, `teams`, `classes`, `pos`), přičemž hned pod smyčkou stejně přenáší `self.pos[k_t].cpu().numpy()` na CPU.

---

## 2. Cílová architektura (Vektorizace na GPU + dávkový přenos telemetrie)

1. **Čistá hranice zámku (`step` vs. interní metody):**
   * `step()` (a všechny veřejné mutátory) drží `with self._engine_lock:`.
   * Interní metody `_step_internal()` a `_update_gates()` předpokládají, že zámek již drží volající.
2. **Plně vektorizované obsazování a bodování uzlů bez větvení na CPU (`(K, M)` matice):**
   * Místo iterace přes `num_nodes` spočítáme matici vzdáleností `dists_sq` tvaru `(K, M)` (kde `K` je počet živých jednotek a `M` počet uzlů) přes broadcasting.
   * Pomocí `torch.nn.functional.one_hot(t_alive, num_classes=4)` a maticového násobení získáme počty jednotek každé frakce ve všech uzlech najednou (`cnts` tvaru `(M, 4)`).
   * Veškeré aktualizace stavu uzlů (`n_owner`, `n_cap`), bonusy pro pěšáky (`CLASS_GRUNT`, `+250 %`, tj. `2.5x`), pasivní týmové body (`+0.08 * step_scale`) i obranná fitness (`+0.25 * step_scale`) proběhnou vektorově na GPU bez jediného `.item()` nebo `.any()`.
   * Kontrola na CPU (`if captured_mask.any():`) zůstane výhradně pro vzácnou událost změny vlastníka uzlu (zápis textu do `self.combat_events`).
3. **GPU akumulace killů a dávkový přenos dat pro telemetrii:**
   * Aktualizace `self.unit_kills`, `self.kills_per_faction` a `self.fitness` proběhne čistě na GPU přes `index_add_` a `torch.bincount`.
   * Pro záznam do `telemetry_mgr` a částice ve `visual_mode` se data pro `k_t` stáhnou na CPU jedním hromadným přenosem (`.cpu().numpy()`).

---

## 3. Detailní kroky implementace v `src/engine_core.py`

### Krok 3.1: Odstranění duplicitních zámků v interních metodách (řádky 504–510 a 1254–1257)

* **Úprava:** Odebrat `with self._engine_lock:` uvnitř `_step_internal()` a `_update_gates()`.
  ```python
  def step(self, dt: float = 0.0166):
      with self._engine_lock:
          self._step_internal(dt)

  def _step_internal(self, dt: float = 0.0166):
      self.operation_time_sec += dt
      self.tick_counter += 1
      # ...

  def _update_gates(self, dt: float = 0.0166):
      """Obléhání zničitelných bran: delegováno do ObjectivesSystem."""
      gates = getattr(self.scenario, "gates", [])
      # ...
  ```
* **Kontrola volajících:** Ověřit v celém repozitáři (`_step_internal`, `_update_gates`), že žádný externí modul nevolá tyto metody napřímo bez `with engine._engine_lock:`.

---

### Krok 3.2: Plná vektorizace obsazování a držení uzlů (řádky 1200–1252)

* **Nahrazení smyčky `for i in range(num_nodes):` maticovým výpočtem na GPU:**
  ```python
  t_alive = self.teams[alive_idx].long()
  step_scale = 6.0
  num_nodes = self.nodes_gpu.shape[0]

  if num_nodes > 0 and K > 0:
      # 1. Maticový výpočet přítomnosti jednotek v uzlech: tvar (K, M)
      diff = p_alive.unsqueeze(1) - self.nodes_gpu[:, :2].unsqueeze(0)   # (K, M, 2)
      dists_sq = (diff * diff).sum(dim=-1)                               # (K, M)
      radii_sq = self.nodes_gpu[:, 2].unsqueeze(0) ** 2                  # (1, M)
      in_node = dists_sq <= radii_sq                                     # (K, M) bool

      # 2. Počty jednotek jednotlivých frakcí v každém uzlu: tvar (M, 4)
      team_one_hot = torch.nn.functional.one_hot(t_alive, num_classes=4).to(self.dtype)
      cnts = in_node.to(self.dtype).transpose(0, 1) @ team_one_hot       # (M, 4)

      cnt_dom, dom_t = torch.max(cnts, dim=1)                            # (M,), (M,)
      cnt_other = cnts.sum(dim=1) - cnt_dom                              # (M,)
      has_majority = cnt_dom > cnt_other                                 # (M,) bool

      n_owner = self.nodes_gpu[:, 3].long()                              # (M,)
      n_cap = self.nodes_gpu[:, 4].clone()                               # (M,)

      # 3. Vektorový výpočet fitness pro obsazující jednotky (včetně +250% bonusu pro CLASS_GRUNT)
      is_grunt = (self.classes[alive_idx] == CLASS_GRUNT)
      bonus_mult = torch.where(is_grunt, 2.5, 1.0).unsqueeze(1)          # (K, 1)

      is_cap_unit = in_node & has_majority.unsqueeze(0) & (t_alive.unsqueeze(1) == dom_t.unsqueeze(0))

      same_owner = (n_owner == dom_t)                                    # (M,)
      reinforcing = has_majority & same_owner & (n_cap < 1.0)            # (M,)
      neutralizing = has_majority & (~same_owner)                        # (M,)

      node_fit_rate = torch.where(
          reinforcing,
          0.45 * step_scale,
          torch.where(neutralizing, 0.65 * step_scale, 0.0)
      ).unsqueeze(0)                                                     # (1, M)

      cap_fitness_gain = (is_cap_unit.to(self.dtype) * bonus_mult * node_fit_rate).sum(dim=1)  # (K,)
      self.fitness[alive_idx] += cap_fitness_gain

      # 4. Aktualizace progresu obsazení (n_cap) a vlastníka (n_owner) na GPU
      n_cap = torch.where(reinforcing, torch.clamp(n_cap + 0.04 * step_scale, max=1.0), n_cap)
      n_cap = torch.where(neutralizing, n_cap - 0.05 * step_scale, n_cap)

      captured_mask = neutralizing & (n_cap <= 0.0)                      # (M,) bool
      n_owner = torch.where(captured_mask, dom_t, n_owner)
      n_cap = torch.where(captured_mask, torch.full_like(n_cap, 0.12), n_cap)

      self.nodes_gpu[:, 3] = n_owner.to(self.dtype)
      self.nodes_gpu[:, 4] = n_cap

      # 5. Zápis události do combat_events pouze ve vzácném tiku změny vlastníka
      if captured_mask.any():
          cap_indices = torch.where(captured_mask)[0].cpu().tolist()
          dom_teams_cpu = dom_t[captured_mask].cpu().tolist()
          sec_t = int(self.operation_time_sec)
          meta_names = ("ALPHA", "OMEGA", "SOLAR", "PHANTOM")
          node_names = ("UZEL ALFA", "UZEL BRAVO", "UZEL CHARLIE", "UZEL DELTA")

          for idx_node, d_team in zip(cap_indices, dom_teams_cpu):
              if len(self.combat_events) > 35:
                  self.combat_events.pop(0)
              n_label = node_names[idx_node] if idx_node < 4 else f"UZEL #{idx_node + 1}"
              self.combat_events.append({
                  "time": self.operation_time_sec,
                  "text": f"[{sec_t:02d}s] {meta_names[d_team]} obsadil {n_label}!",
                  "team": d_team,
              })

      # 6. Pasivní bodování a obranná fitness bez CPU větvení (n_owner >= 0 a n_cap >= 0.7)
      held_mask = (n_owner >= 0) & (n_cap >= 0.7)                        # (M,) bool
      safe_owners = torch.clamp(n_owner, min=0)
      pts_add = torch.where(held_mask, 0.08 * step_scale, 0.0)
      self.node_control_points.index_add_(0, safe_owners, pts_add)

      is_defending = in_node & held_mask.unsqueeze(0) & (t_alive.unsqueeze(1) == n_owner.unsqueeze(0))
      def_nodes_count = is_defending.sum(dim=1).to(self.dtype)           # (K,)
      self.fitness[alive_idx] += def_nodes_count * (0.25 * step_scale)
  ```

---

### Krok 3.3: GPU akumulace a dávkový přenos pro killy (řádky 1340–1375)

* **Nahrazení opakovaných `.item()` jedním hromadným přenosem a GPU operací:**
  ```python
  shooters = buf.last_attacker_id[k_t].long()
  valid_mask = (shooters >= 0) & (shooters < self.N)
  valid_shooters = shooters[valid_mask]

  if valid_shooters.numel() > 0:
      ones = torch.ones_like(valid_shooters, dtype=self.unit_kills.dtype)
      self.unit_kills.index_add_(0, valid_shooters, ones)
      self.fitness.index_add_(
          0,
          valid_shooters,
          torch.full_like(valid_shooters, 30.0, dtype=self.fitness.dtype),
      )

      killer_teams = self.teams[valid_shooters].long()
      faction_kills = torch.bincount(killer_teams, minlength=4)
      self.kills_per_faction[:4] += faction_kills[:4]

  need_cpu_kill_data = (self.telemetry_mgr is not None) or self.visual_mode
  if need_cpu_kill_data:
      v_ids_cpu = k_t.cpu().numpy()
      pos_k = self.pos[k_t].cpu().numpy()
      teams_k = self.teams[k_t].cpu().numpy()
      classes_k = self.classes[k_t].cpu().numpy()

      if self.telemetry_mgr is not None:
          shooters_cpu = shooters.cpu().numpy()
          all_teams_cpu = self.teams.cpu().numpy()
          all_classes_cpu = self.classes.cpu().numpy()

          for ki in range(len(v_ids_cpu)):
              s_id = int(shooters_cpu[ki])
              if 0 <= s_id < self.N:
                  self.telemetry_mgr.record_kill_event(
                      epoch=self.current_epoch,
                      time_sec=self.operation_time_sec,
                      killer_team=int(all_teams_cpu[s_id]),
                      victim_team=int(teams_k[ki]),
                      killer_class=int(all_classes_cpu[s_id]),
                      victim_class=int(classes_k[ki]),
                      x=float(pos_k[ki, 0]),
                      y=float(pos_k[ki, 1]),
                  )
  ```

4. Akceptační kritéria a ověřovací testy (Checklist)

• Nulový výskyt .item() v horké smyčce: Statická kontrola v metodě _step_internal() (včetně sekcí pro uzly a killy) vrátí 0 nálezů .item() uvnitř for cyklů.
• Deterministická shoda herní logiky: Rychlost obsazování uzlů (včetně 2,5× bonusu pro CLASS_GRUNT), přepínání vlastnictví uzlů, zápis do combat_events, pasivní bodování i přičítání +30.0 fitness za kill dává na stejném seedu totožné výsledky jako původní implementace.
• Měřitelné zrychlení tiku: Benchmark 1 000 tiků s 800 jednotkami na GPU (v headless režimu) vykáže zkrácení průměrného času _step_internal() minimálně o 20–30 %.
• Bezpečnost vláken: Testovací sada tests/test_suite.py a souběžný běh s Fází 1 (ukládání snapshotů za běhu) projdou bez deadlocku a bez regrese.



