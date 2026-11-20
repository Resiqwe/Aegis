# JAK VOJÁK VNÍMÁ SVĚT A JAK ROZHODUJE MOZEK (KÓDOVÁ UKÁZKA)

Tento dokument vysvětluje konkrétní fragmenty kódu z jádra simulátoru **Cyber-Tactics**, které ukazují:
1. **Jak se fyzikální realita bojiště přepočítává do smyslů vojáka** (egocentrické senzory).
2. **Jak neuronová síť bez jediné podmínky `if/else` tyto vjemy vyhodnotí do 7 tělesných akcí**.

---

## 1. Část: Jak voják vnímá svět (Senzory)
*Soubor:* `src/engine_core.py` (řádky 846–908)

Voják nemá přístup k mapovým souřadnicím `[x, y]` celé herní plochy. Vnímá svět **ze svého pohledu** (tzv. egocentrická soustava – *přede mnou / za mnou / vpravo / vlevo*).

Kód vezme úhel natočení vojáka (`self.heading`) a pomocí goniometrických funkcí (kosinus a sinus) otočí globální světové vektory do jeho zorného pole:

```python
# ==============================================================
# EGOCENTRICKÁ TRANSFORMACE SENZORŮ DO SOUSTAVY TĚLA VOJÁKA
# ==============================================================
cos_th = torch.cos(self.heading)
sin_th = torch.sin(self.heading)

# 1. Kde je nepřítel vůči mé hrudi a očím? (+1 vpředu, -1 vzadu / +1 vpravo, -1 vlevo)
enemy_fwd   =  dir_enemy[:, 0] * cos_th + dir_enemy[:, 1] * sin_th
enemy_right = -dir_enemy[:, 0] * sin_th + dir_enemy[:, 1] * cos_th

# 2. Odkud po mně právě střílí? (směrový vektor příchozí palby)
fire_fwd    =  self.incoming_fire_dir[:, 0] * cos_th + self.incoming_fire_dir[:, 1] * sin_th
fire_right  = -self.incoming_fire_dir[:, 0] * sin_th + self.incoming_fire_dir[:, 1] * cos_th

# 3. Kde je taktický rozkaz velitele (kam má četa postupovat)?
order_fwd   =  dir_order[:, 0] * cos_th + dir_order[:, 1] * sin_th
order_right = -dir_order[:, 0] * sin_th + dir_order[:, 1] * cos_th

# 4. Kde je nejbližší stěna / překážka, za kterou se můžu schovat?
cover_fwd   =  cover_dir[:, 0] * cos_th + cover_dir[:, 1] * sin_th
cover_right = -cover_dir[:, 0] * sin_th + cover_dir[:, 1] * cos_th

# ==============================================================
# SESTAVENÍ TENZORU 14 VSTUPNÍCH SMYSLŮ (SENSORS)
# ==============================================================
sensors[:, 0]  = enemy_fwd                                           # Směr k cíli: Vpřed / Vzad (-1..+1)
sensors[:, 1]  = enemy_right                                         # Směr k cíli: Vpravo / Vlevo (-1..+1)
sensors[:, 2]  = (1.0 - (dist_enemy / self.class_range)).clamp(-1, 1)# Vzdálenost k cíli (v dostřel u zbraně?)
sensors[:, 3]  = fire_fwd                                            # Příchozí střely: Vpřed / Vzad
sensors[:, 4]  = fire_right                                          # Příchozí střely: Vpravo / Vlevo
sensors[:, 5]  = order_fwd                                           # Rozkaz generála: Vpřed / Vzad
sensors[:, 6]  = order_right                                         # Rozkaz generála: Vpravo / Vlevo
sensors[:, 7]  = self.suppression                                    # Šok / panika (míra potlačení palbou: 0..1)
sensors[:, 8]  = (self.ammo_in_mag / self.class_mag_size) * 2.0 - 1.0# Zůstatek nábojů v zásobníku (-1..+1)
sensors[:, 9]  = (self.hp / self.class_max_hp) * 2.0 - 1.0           # Zdraví těla (-1 prázdné .. +1 plné)
sensors[:, 10] = cover_fwd                                           # Kryt: Vpřed / Vzad
sensors[:, 11] = cover_right                                         # Kryt: Vpravo / Vlevo
sensors[:, 12] = cohesion                                            # Koheze: jsou parťáci blízko (-1 sám .. +1 četa)?
sensors[:, 13] = zone_status                                         # Stav zóny (-1 nepřítel .. +1 naše zóna)
```

---

## 2. Část: Jak to mozek vyhodnocuje (Neuronový výpočet)
*Soubor:* `src/ai_brains.py` (řádky 81–115)

Těchto **14 čísel** vstupuje do digitálního mozku. Všimněte si, že **zde není ani jedna podmínka `if`**. Celé rozhodování probíhá jako maticové násobení vah (`W1`, `W2`, `W3`) na grafické kartě a nelineární aktivace (`tanh` a `sigmoid`):

```python
def tactical_brain_forward(
    x: torch.Tensor,
    W1: torch.Tensor, b1: torch.Tensor,
    W2: torch.Tensor, b2: torch.Tensor,
    W3: torch.Tensor, b3: torch.Tensor,
) -> torch.Tensor:
    """
    x: Tenzor 14 vjemů vojáka (Batch, 14)
    W1, W2, W3: Matice vah (synapse v mozku vyvíjené evolucí)
    b1, b2, b3: Prahové bias neuronů
    """
    x_3d = x.unsqueeze(1)
    
    # -------------------------------------------------------------
    # 1. Vrstva: Percepce a okamžité reflexy (14 vjemů -> 64 signálů)
    # -------------------------------------------------------------
    # Tady se v mozku aktivují základní pocity:
    # "Hrozí nebezpečí", "Mám zaměřeno", "Dochází munice"
    h1 = torch.tanh(torch.bmm(x_3d, W1) + b1)
    
    # -------------------------------------------------------------
    # 2. Vrstva: Taktická syntéza (64 signálů -> 64 signálů)
    # -------------------------------------------------------------
    # Zde se pocity zkombinují do plánu:
    # "Pokud po mně pálí a nemám náboje, kašli na útok a okamžitě hledej kryt"
    h2 = torch.tanh(torch.bmm(h1, W2) + b2)
    
    # -------------------------------------------------------------
    # 3. Výstupní vrstva: Motorické povely tělu (64 signálů -> 7 akcí)
    # -------------------------------------------------------------
    raw_out = torch.bmm(h2, W3) + b3
    
    # Převedení výstupních hodnot na fyzický pohyb a akce:
    
    # [0, 1, 2] = Pohyb těla v rozsahu -1.0 až +1.0:
    # 0: Surge (Plyn dopředu / Couvání vzad)
    # 1: Sway (Úkrok / Strafe vpravo či vlevo)
    # 2: Yaw (Otáčení těla a zbraně nezávisle na směru běhu)
    surge_sway_yaw = torch.tanh(raw_out[:, :, 0:3])
    
    # [3] = Spoušť zbraně: Sigmoid převede signál na pravděpodobnost 0.0 až 1.0 (> 0.5 = střelba)
    trigger = torch.sigmoid(raw_out[:, :, 3:4])
    
    # [4] = Postoj (Stance): +1.0 Sprint přes pole / -1.0 Plížení a zalehnutí v krytu
    stance = torch.tanh(raw_out[:, :, 4:5])
    
    # [5] = Taktická dovednost: > 0.60 aktivuje hod dýmovnice, léčebný paprsek nebo štít
    ability = torch.sigmoid(raw_out[:, :, 5:6])
    
    # [6] = Komunikační signál vysílaný spojencům v četě
    ping = torch.tanh(raw_out[:, :, 6:7])
    
    # Složení výsledného 7-prvkového vektoru akce
    out = torch.cat([surge_sway_yaw, trigger, stance, ability, ping], dim=2)
    return out.squeeze(1)
```

---

## 3. Proč to funguje lépe než skriptování podmínek?
1. **Plynulost a holonomie:** Voják se nerozhoduje binárně (např. *stůj* nebo *jdi*). Mozek mu pošle spojité hodnoty – např. `Surge = 0.75` (rychlý krok vpřed) a zároveň `Sway = -0.40` (úkrok vlevo), takže se pohybuje přirozeně po diagonále, kličkuje a zároveň míří na cíl jiným směrem (`Yaw`).
2. **Přirozená emergence taktiky:** Všechna čísla v maticích `W1`, `W2`, `W3` se ladí evolucí. Pokud voják náhodnou mutací získal propojení *„mám málo HP + střílí po mně $\rightarrow$ běž k nejbližšímu krytu a hoď dým“*, přežil déle a způsobil vyšší poškození. Tím získal vysokou fitness a jeho mozek se zkopíroval do dalších generací, zatímco hloupí vojáci vyhynuli.
