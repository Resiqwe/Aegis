import matplotlib.pyplot as plt

ucinne_zaznamy = []
aktualni_beh = []
ima_casovy_sloupec = False

with open("tréninkový_log.txt", "r", encoding="utf-8") as f:
    for radek in f:
        # Detekce nového běhu
        if "NOVÝ TRÉNINKOVÝ BĚH" in radek or "BĚH S MĚŘENÍM ČASU" in radek or "ROZHODOVACÍ BĚH" in radek:
            if aktualni_beh:
                ucinne_zaznamy.append(aktualni_beh)
                aktualni_beh = []
            ima_casovy_sloupec = False
            continue
        
        # Pokud řádek obsahuje hlavičkový sloupec "Čas (s)", víme, jak tabulku číst
        if "Čas (s)" in radek:
            ima_casovy_sloupec = True
            continue
            
        # Zpracování řádků s daty
        hodnoty = radek.split()
        if len(hodnoty) >= 2:
            try:
                e = float(hodnoty[0])
                # Pokud log obsahuje čas, je loss na 3. pozici (index 2), jinak na 2. pozici (index 1)
                loss_index = 2 if ima_casovy_sloupec else 1
                
                if len(hodnoty) > loss_index:
                    l = float(hodnoty[loss_index])
                    aktualni_beh.append({"epoch": e, "loss": l})
            except ValueError:
                continue

if aktualni_beh:
    ucinne_zaznamy.append(aktualni_beh)

print(f"Celkem načteno běhů v logu: {len(ucinne_zaznamy)}")

# Deduplikace
unikatni_behy = []
zpracovane_signatury = set()

for beh in ucinne_zaznamy:
    if not beh:
        continue
    signatura = (beh[0]["loss"], beh[-1]["loss"], len(beh))
    if signatura not in zpracovane_signatury:
        zpracovane_signatury.add(signatura)
        unikatni_behy.append(beh)

print(f"Počet unikátních běhů po odfiltrování duplikátů: {len(unikatni_behy)}")

# Vizualizace v grafu
plt.figure(figsize=(10, 6))

for i, beh in enumerate(unikatni_behy):
    epochy = [z["epoch"] for z in beh]
    lossy = [z["loss"] for z in beh]
    plt.plot(epochy, lossy, marker='o', linestyle='-', label=f"Unikátní experiment #{i+1}")

plt.title("Srovnání unikátních tréninkových běhů (Pokles Loss)")
plt.xlabel("Epocha")
plt.ylabel("Hodnota Loss (MSE)")
plt.grid(True)
plt.legend()
plt.show()