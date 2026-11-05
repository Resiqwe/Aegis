# PLAN: sběr bodů (zatím NEimplementovat)

Postup: nápady se jen zapisují. Po povelu „sestav plán" se seřadí podle závislostí a po „schvaluju" se provedou najednou s jedním závěrečným hlášením.

## Cíl projektu
Naučit se vytvářet neuronové sítě a hrát si s AI (Python, CUDA, PyTorch). Učení a experimenty, od jednoduchých věcí ke složitějším.

## Ověřený stav
- `python run_all.py` projde celou trubku (trénink, dump, validace, analýza, zpětná kontrola uložených vah). Při chybě v kterémkoli kroku se zastaví s exit code 1 (ověřeno podstrčeným poškozeným dumpem).
- Dump ukládá skutečné váhy (W jako čísla, bias a pravděpodobnosti v meta JSON). Staré neplatné dumpy jsou v `dumps/_stare_neplatne/`.

## Nasbírané body
1. [x] Nainstalovat PyTorch s CUDA do `.venv` a ověřit, že vidí GPU
2. [x] První síť od nuly: jeden neuron a gradientní sestup (bez frameworku)
3. [ ] Klasifikace číslic MNIST (malá síť, trénink na GPU)

## Pravidla dávky
- Krátká kontrola po každém kroku, ať se chyba nenese dál.
- Jedno závěrečné hlášení s tabulkou skutečných čísel.
- Mazání a změny gitu schvaluje člověk.
