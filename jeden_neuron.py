import numpy as np
import time

# 1. Definice Sigmoidy (náš "rozhodovací filtr / razítko")
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Příprava dat pro rozhodování
X = np.array([
    [1.0, 2.0, 1.0],
    [2.0, 1.0, 2.0],
    [3.0, 2.0, 0.0],
    [4.0, 3.0, 1.0]
])

y = np.array([0.0, 0.0, 1.0, 1.0])

# 3. Inicializace vah a biasu
np.random.seed(42)
W = np.random.randn(3) 
b = 0.0                

print(f"Počáteční náhodné váhy W: {W}")

# 4. Tréninkový proces pro rozhodování
learning_rate = 0.1   
max_cas_sekundy = 2.0  

logy_tréninku = []
start_cas = time.time()
epoch = 0

print(f"\nSpouštím trénink rozhodovacího bloba...")

while True:
    aktualni_cas = time.time() - start_cas
    if aktualni_cas >= max_cas_sekundy:
        break
        
    epoch += 1
    
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)
    loss = np.mean((y_pred - y) ** 2)
    
    error = y_pred - y
    grad_W = np.dot(X.T, error * y_pred * (1 - y_pred)) * (2 / len(X))
    grad_b = np.mean(2 * error * y_pred * (1 - y_pred))
    
    W = W - learning_rate * grad_W
    b = b - learning_rate * grad_b
    
    # Logování každých 5000 epoch nebo na konci
    if epoch % 5000 == 0:
        logy_tréninku.append({
            "epoch": epoch,
            "loss": loss,
            "weights": W.copy(),
            "bias": b,
            "time": aktualni_cas
        })

celkovy_cas = time.time() - start_cas

print(f"\n--- Výsledek rozhodování ---")
print(f"Celkový reálný čas běhu: {celkovy_cas:.4f} sekund (Celkem epoch: {epoch})")
print(f"Naučené váhy W: {W}")
print(f"Naučený bias b: {b}")

# Zápis logů do souboru (tohle předtím chybělo)
if len(logy_tréninku) > 0:
    with open("tréninkový_log.txt", "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 10 + f" ROZHODOVACÍ BĚH (Klasifikace, {celkovy_cas:.3f}s) " + "=" * 10 + "\n")
        f.write("Epocha\tČas (s)\t\tLoss\t\t\tVáhy W\t\t\t\tBias\n")
        f.write("-" * 85 + "\n")
        for záznam in logy_tréninku:
            w_str = f"[{záznam['weights'][0]:12.4f}, {záznam['weights'][1]:12.4f}, {záznam['weights'][2]:12.4f}]"
            f.write(f"{záznam['epoch']:<6}\t{záznam['time']:<10.4f}\t{záznam['loss']:<15.6f}\t{w_str}\t{záznam['bias']:.4f}\n")
    print("Logy byly úspěšně zapsány do souboru 'tréninkový_log.txt'!")
else:
    print("Upozornění: Trénink byl příliš krátký a nevytvořil žádné logy.")

# Zkouška po tréninku
z_test = np.dot(X, W) + b
vysledne_pravdepodobnosti = sigmoid(z_test)
print(f"\nVstupy: \n{X}")
print(f"Výsledné pravděpodobnosti od bloba: {vysledne_pravdepodobnosti}")
print(f"Zaokrouhlená rozhodnutí (0 nebo 1): {np.round(vysledne_pravdepodobnosti)}")
print(f"Cílová správná odpověď:             {y}")