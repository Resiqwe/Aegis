import json
import numpy as np
import time


class NeuronCore:
    def __init__(self, input_dim=3, learning_rate=0.1):
        # --- 2. MODEL (Jádro / Tranzistor) ---
        np.random.seed(42)  # Pro reprodukovatelnost v naší "lebce"
        self.W = np.random.randn(input_dim)
        self.b = 0.0
        self.lr = learning_rate

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def step(self, X, y):
        """Provede jeden tréninkový krok (epochu) uvnitř lebky."""
        z = np.dot(X, self.W) + self.b
        y_pred = self._sigmoid(z)
        loss = np.mean((y_pred - y) ** 2)

        # Zpětná propagace (matematický gradient)
        error = y_pred - y
        grad_W = np.dot(X.T, error * y_pred * (1 - y_pred)) * (2 / len(X))
        grad_b = np.mean(2 * error * y_pred * (1 - y_pred))

        # Aktualizace vah a biasu
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b

        return loss, y_pred

    def predict(self, X):
        """Vyvěsí finální rozhodnutí (razítko 0 nebo 1)."""
        z = np.dot(X, self.W) + self.b
        pravdepodobnosti = self._sigmoid(z)
        return pravdepodobnosti, np.round(pravdepodobnosti)


# --- 1. INSTRUKČNÍ VSTUP (Environment / Realita krabičky) ---
# Definujeme jednoduchá data pro naše logické rozhodování
X_env = np.array([
    [1.0, 2.0, 1.0],
    [2.0, 1.0, 2.0],
    [3.0, 2.0, 0.0],
    [4.0, 3.0, 1.0]
])
y_env = np.array([0.0, 0.0, 1.0, 1.0])


def main():
    # Inicializace core v naší lebce
    core = NeuronCore(input_dim=3, learning_rate=0.1)

    # --- 3. DIAG PORT (Telemetrie a diagnostika) ---
    print("=== [DIAG PORT] Start kybernetického core ===")
    print(f"Počáteční stav vah W: {core.W}")
    print(f"Počáteční bias b: {core.b}\n")

    max_cas_sekund = 2.0
    start_cas = time.time()
    epoch = 0
    loss = float("nan")

    print("Spouštím uzavřenou smyčku v lebce...")
    while True:
        aktualni_cas = time.time() - start_cas
        if aktualni_cas >= max_cas_sekund:
            break

        epoch += 1
        loss, _ = core.step(X_env, y_env)

        # Diag výpis každých 50 000 epoch
        if epoch % 50000 == 0:
            print(f"[DIAG] Epocha: {epoch:<7} | Čas: {aktualni_cas:.3f}s | Loss: {loss:.6f}")

    # Závěrečná diagnostika po doběhnutí limitu
    pravdepodobnosti, rozhodnuti = core.predict(X_env)

    print("\n=== [DIAG PORT] Konec cyklu ===")
    print(f"Celkem odtočeno epoch: {epoch}")
    print(f"Finální váhy W: {core.W}")
    print(f"Finální bias b: {core.b}")
    print(f"Vstupy:\n{X_env}")
    print(f"Výsledné pravděpodobnosti: {pravdepodobnosti}")
    print(f"Rozhodnutí krabičky (0/1): {rozhodnuti}")
    print(f"Očekávaná realita (Cíl):   {y_env}")

    # Strojově čitelný stav pro pipeline (místo hledání v textu).
    stav = {
        "epochs": epoch,
        "loss": float(loss),
        "weights": [float(w) for w in core.W],
        "bias": float(core.b),
        "probabilities": [float(p) for p in pravdepodobnosti],
    }
    print("STATE_JSON: " + json.dumps(stav))


if __name__ == "__main__":
    main()
