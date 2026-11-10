# PLAN

Postup: nápady se jen zapisují. Po povelu "sestav plán" se seřadí podle závislostí a po "schvaluju" se provedou najednou s jedním závěrečným hlášením.

## Cíl projektu
Stavět výpočetní logiku z neuronů po vrstvách (neuron, hradlo, blok, jádro, čiplet, čip) a zachytit "evoluci": porovnat generace podle počtu neuronů, času učení a přesnosti. Vše ověřovat vyčerpávajícím testem.

## Hotovo
- [x] `perceptron.py`: perceptron, hradla, sčítání a odčítání, SR latch (samokontrola projde)
- [x] `brick.py`: první kostička, síť učená celým blokem (AND, NOR, XOR, plná sčítačka, 2bitová sčítačka)
- [x] Úklid struktury a dokumentace, starou linii ponechat jako dědictví

## Fáze B: evoluce bloků
1. [ ] Měření "tabulka vs. síť": sčítačky 1 až 4 bity jako jedna kostička (neurony, parametry, čas učení)
2. [ ] Skládání: 4bitová sčítačka z plných sčítaček vs. naučená naráz
3. [ ] Hledání nejmenší sítě pro daný blok (základ evoluce architektur)

## Fáze C: k procesoru
4. [ ] Registr a paměť z `SRLatch`
5. [ ] ALU z kostiček
6. [ ] Jednoduché jádro bez zkratek (každá operace přes neurony)

## Odbočky (až bude základ)
- Šum a zpoždění v kostičkách, měření chyb při vyšší frekvenci
- Automat na hledání chyb a záplat (pozorovatel, hledač příčin, návrhář záplat, ověřovatel)
- Interaktivní učebnice

## Technické dluhy staré linie
- `demos/cpu_fibonacci_long_run.py` překrývá program s daty a nepočítá Fibonacciho čísla
- `torch` a `matplotlib` v `requirements.txt` kód nepoužívá
- Duplicity: dvě CPU, dvě ALU, kalibrace bran na 4 místech
- `neuron/chain_runner.py` nikdo neimportuje

## Pravidla dávky
- Krátká kontrola po každém kroku.
- Jedno závěrečné hlášení s tabulkou skutečných čísel.
- Mazání a změny gitu schvaluje člověk.
