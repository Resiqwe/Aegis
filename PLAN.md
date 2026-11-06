# PLAN

Postup: nápady se jen zapisují. Po povelu "sestav plán" se seřadí podle závislostí a po "schvaluju" se provedou najednou s jedním závěrečným hlášením.

## Cíl projektu
Naučit se stavět výpočetní logiku z neuronů: od jedné buňky přes ALU a registry po CPU s hodinami a podmíněnými skoky.

## Hotovo (podle kódu, testy neověřeny)
- [x] Základní buňka `neuron_node` s dumpy, hash řetězem, WORM a auditem
- [x] Řetězení buněk (`chain_runner`)
- [x] CPU step1-4 (XOR, ALU, registry, jádro), telemetrie, hodiny, Zero Flag, JZ/JNZ
- [x] Demo programy (Fibonacci, Gauss, 8bit ADC)
- [x] Přestavba na balíčky `neuron/ cpu/ demos/ tests/`

## Otevřené body (technické dluhy)
1. [ ] Spustit a vyhodnotit obě testovací sady po přesunu
2. [ ] Jednotný spouštěč (náhrada za dřívější `run_all.py`)
3. [ ] Ověřit překryv `cpu_telemetry_probe` a `demos/cpu_system_telemetry`
4. [ ] Rozhodnout o `torch` a `matplotlib` v `requirements.txt` (kód používá jen NumPy)
5. [ ] Rozhodnout o osudu `archiv/` a starých `dumps/`

## Pravidla dávky
- Krátká kontrola po každém kroku.
- Jedno závěrečné hlášení s tabulkou skutečných čísel.
- Mazání a změny gitu schvaluje člověk.