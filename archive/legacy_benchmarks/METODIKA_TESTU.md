# Metodika testování buňky (`neuron_node.py`)

Spouštěč testovací sady: [`test_torture_suite.py`](file:///c:/Users/resiq/Documents/projekty/pokec/test/tests/test_torture_suite.py)  
Testovaný objekt (černá skříňka – do jejího kódu se během testů nesahá): [`neuron_node.py`](file:///c:/Users/resiq/Documents/projekty/pokec/test/neuron/neuron_node.py)

---

## Základní principy metodiky

1. **Izolace prostředí (Sandbox):** Testovací sada si před spuštěním zálohuje obsah `dumps/` (pokud existuje), každou skupinu testů spouští nad čistým stavem a po skončení zanechá pouze ověřené artefakty nebo detailní zprávu o selhání.
2. **Nedotknutelnost buňky:** Samotný soubor `neuron_node.py` se nemodifikuje. Všechny útoky probíhají buď přes veřejné vstupy (`X`, `y`, parametry volání), nebo přímou manipulací se soubory na disku v přirozeném gapu mezi takty.
3. **Dvojí verifikace (Pozitivní vs. Negativní testy):**
   - **Kouřové a zátěžové testy** musí projít bez chyby (`PASS` = žádná výjimka, matematická shoda).
   - **Sabotážní testy (zvenčí i zevnitř)** musí buňka **odhalit a hlasitě odmítnout** (`PASS` = buňka vyhodila očekávanou výjimku `RuntimeError` / `ValueError` a odmítla kontaminovaná data). Pokud by buňka sabotáž spolkla potichu, test končí jako `FAIL`.

---

## Kategorie 1: Klasické kouřáky (Smoke Tests)
Ověřují, že základní životní cyklus buňky (studený start -> trénink -> dump -> validace -> warm-start -> zpětná kontrola) funguje za normálních podmínek.

| ID | Název testu | Co přesně dělá | Kritérium úspěchu (`PASS`) |
|---|---|---|---|
| **S0** | **Subprocess CLI spuštění (Black-box OS proces)** | Spustí `python neuron_node.py` jako čerstvý samostatný proces přes `subprocess.run`. | Návratový kód `0`, na disku 3 páry `.dat`/`.json`, 3 záznamy v `node_events.jsonl`, `0` souborů `.tmp`. |
| **S1** | **Studený start a Early-Exit** | Spustí jeden takt `train_fast` na standardních datech s cílem `target_loss=1e-4`. | `stop_reason == "EARLY_EXIT"`, `mse_loss <= 1e-4`, latence `< 50 ms`. |
| **S2** | **Integrita artefaktů (.dat + .json + .jsonl)** | Uloží takt přes `extract_and_dump` a zkontroluje soubory na disku. | Existuje `_weights.dat` (4 řádky = 3 váhy + bias), `_meta.json` (platný JSON s `weights_hash`) a v `node_events.jsonl` přibyl záznam. Žádné `.tmp` soubory. |
| **S3** | **Bitová zpětná kontrola (Round-trip)** | Načte uložený `.dat` do nové prázdné instance a porovná predikci. | `np.allclose` s přesností `1e-12` vůči `probabilities` v občance; zaokrouhlená rozhodnutí přesně `[0, 0, 1, 1]`. |
| **S4** | **Třídobý Warm-Start (předání štafety)** | Druhá instance se probudí přes `load_from_verified_dump` z dumpu první instance a pokračuje na přísnější `target_loss=1e-5`. | `total_epochs` se sčítá, `parent_weights_hash` v 2. občance se rovná `weights_hash` 1. občanky, chyba klesla pod `1e-5`. |
| **S5** | **Bitový determinismus při stejném seedu** | Dva nezávislé běhy se `seed=42` a stejným cílem. | Oba běhy vygenerují na bit identický `weights_hash`. |

---

## Kategorie 2: Zátěžové testy (Stress & Endurance)
Ověřují chování buňky pod vysokou kadencí, při neřešitelných úlohách a při změně reality (online přeučování).

| ID | Název testu | Co přesně dělá | Kritérium úspěchu (`PASS`) |
|---|---|---|---|
| **Z1** | **Rychlopalba 50 taktů (Zero-Sleep Burst)** | Spustí 50 navazujících taktů v těsné smyčce bez jakéhokoliv `sleep()`. | Vznikne přesně 50 unikátních párů souborů (žádná kolize `timestamp_us`), kompletní souvislý řetěz `parent_weights_hash` od 1 do 50, žádné osiřelé `.tmp` soubory. |
| **Z2** | **Watchdog pod neřešitelnou úlohou (XOR)** | Předhodí jednomu lineárnímu neuronu neřešitelný problém XOR (`[0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0`) s `target_loss=1e-6` a časovým stropem `max_cas_sekund=0.025` (25 ms). | Smyčka nezamrzne! Skončí se `stop_reason == "TIME_LIMIT"` v okně `25–45 ms`. |
| **Z3** | **Šoková změna reality (Inverze cílů za chodu)** | Takt 1 natrénuje buňku na `y = [0, 0, 1, 1]`. Takt 2 si natáhne její váhy (warm-start), ale dostane opačnou realitu `y_inv = [1, 1, 0, 0]`. | Buňka dokáže během jediného časového okna překlopit znaménka všech vah a biasu a dosáhnout `100%` shody na nové realitě. |

---

## Kategorie 3: Sabotáž zvenčí (External Tampering / Disk Attacks)
Simuluje útočníka nebo chybu úložiště, která v přirozeném gapu mezi takty modifikuje soubory v `dumps/` předtím, než si je další buňka natáhne přes `load_from_verified_dump` nebo `validate_dumps`.

| ID | Název testu | Vektor útoku | Očekávaná obrana (`PASS`) |
|---|---|---|---|
| **E1** | **Falšování váhy $W_1$ o 1 bit / desetinné místo** | V otevřeném `_weights.dat` změní první řádek o `0.000001`. | `validate_dumps` i `load_from_verified_dump` okamžitě vyhodí `RuntimeError` (nesedí `SHA-256` občanka). |
| **E2** | **Falšování samotného Biasu** | V `_weights.dat` změní pouze poslední (4.) řádek (`bias`). | `validate_dumps` vyhodí `RuntimeError` (důkaz, že bias je nyní pod ochranou `SHA-256`). |
| **E3** | **Útok podvrženou občankou (Trojský kůň v `.dat`)** | Útočník vymění číslo v `_weights.dat` za text `"HACKED"` A ZÁROVEŇ přepočítá a přepíše `weights_hash` v `_meta.json`, aby haš seděl. | Haš sice projde, ale druhá vrstva obrany (parsování `float` ve validátoru) vyhodí `RuntimeError: nejsou platná čísla`. |
| **E4** | **Útok oříznutím parametru + přepis haše** | Útočník smaže řádek biasu v `.dat` (zůstanou jen 3 čísla místo 4) a přepíše `weights_hash` v občance. | Třetí vrstva obrany (`param_count`) vyhodí `RuntimeError: Nesedí počet parametrů`. |
| **E5** | **Smazání souboru vah (osiřelá občanka)** | Smaže `_weights.dat`, ponechá `_meta.json`. | Validátor vyhodí `RuntimeError: Chybí soubor vah`. |
| **E6** | **Poškození JSON občanky (Corrupted Meta)** | Zapíše nevalidní JSON sekvenci do `_meta.json`. | Validátor vyhodí `RuntimeError: Poškozený JSON`. |
| **E7** | **Tichá změna predikce v občance (Round-trip past)** | Útočník změní `probabilities[0]` v `_meta.json` o `0.01` (haš vah přitom pořád sedí!). | `roundtrip_verification` odhalí nesoulad mezi reálným výpočtem z `.dat` a tvrzením v občance a vyhodí `RuntimeError`. |
| **E8** | **Podvržení cizího mozku (Swap Attack)** | Útočník prohodí `_weights.dat` mezi Během #1 a Během #2. | Občanka Běhu #2 odhalí cizí `weights_hash` a vyhodí `RuntimeError`. |

---

## Kategorie 4: Sabotáž zevnitř (Internal Poisoning / Bad Actor Injection)
Simuluje situaci, kdy buňka dostane jedovaté vstupy z nadřazeného vlákna nebo je její vnitřní stav v paměti poškozen těsně před uložením.

| ID | Název testu | Vektor útoku | Očekávaná obrana (`PASS`) |
|---|---|---|---|
| **I1** | **Injekce `NaN` / `Inf` do vah před dumpem** | Vnitřní stav `core.W[0]` je nastaven na `float('nan')`, vyexportuje se a jiná buňka ho zkusí použít v `roundtrip_verification`. | Zpětná kontrola (`np.allclose` / porovnání s cílem) odmítne `NaN` predikci a vyhodí `RuntimeError`. |
| **I2** | **Pokus o Warm-Start do buňky s jinou dimenzí** | Dump ze 3-vstupové buňky (`param_count=4`) se zkusí natáhnout do buňky inicializované s `input_dim=5`. | `load_from_verified_dump` vyhodí `RuntimeError: Očekáváno 6 parametrů, nalezeno 4`. |
| **I3** | **Extrémní vstupy (Saturace Sigmoidy / Overflow)** | Vstupy `X` obsahují hodnoty `1000.0` a `-1000.0`, kde `exp(1000)` hrozí přetečením do `Inf`. | Kontrola, jak se jádro zachová při numerické saturaci a zda nespadne na nekontrolovanou výjimku. |
| **I4** | **Ochrana RAM při otráveném Warm-Startu + Desync RAM** | Buňka se pokusí zavolat `load_from_verified_dump` nad zfalšovaným dumpem; kontroluje se, zda původní `victim.W` v RAM zůstalo nedotčeno. | Metoda vyhodí `RuntimeError` PŘED přepsáním `self.W`, paměť buňky zůstane nekontaminovaná. |
