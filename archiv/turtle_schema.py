"""
turtle_schema.py - Vykreslení architektury našeho kyber-neuronu a L1 orchestrátoru.
Využívá Python knihovnu turtle s retro cyber/industrial dark-mode estetikou.
"""

import turtle
import time

def setup_canvas():
    screen = turtle.Screen()
    screen.setup(width=1300, height=850)
    screen.bgcolor("#0f141c")
    screen.title("SCHÉMA ARCHITEKTURY: Cyber-Neuron & L1 Pipeline Runner (SCADA / ISO)")
    screen.tracer(0)  # Vypneme automatické překreslování pro plynulé skriptování
    return screen

def make_turtle(color="#00ffcc", speed=0):
    t = turtle.Turtle()
    t.hideturtle()
    t.speed(speed)
    t.pensize(2)
    t.color(color)
    t.penup()
    return t

def draw_box(t, x, y, width, height, fill_color, border_color="#00e5ff", title=None):
    t.penup()
    t.goto(x, y)
    t.setheading(0)
    t.color(border_color, fill_color)
    t.pendown()
    t.begin_fill()
    for _ in range(2):
        t.forward(width)
        t.right(90)
        t.forward(height)
        t.right(90)
    t.end_fill()
    t.penup()
    
    if title:
        t.goto(x + 12, y - 22)
        t.color("#e2e8f0")
        t.write(title, font=("Consolas", 11, "bold"))

def draw_arrow(t, start_x, start_y, end_x, end_y, color="#38bdf8", label=None, label_side="top"):
    t.penup()
    t.goto(start_x, start_y)
    t.setheading(t.towards(end_x, end_y))
    t.color(color)
    t.pendown()
    t.pensize(2)
    t.goto(end_x, end_y)
    t.stamp()
    t.penup()
    
    if label:
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        offset_y = 7 if label_side == "top" else -18
        t.goto(mid_x, mid_y + offset_y)
        t.color(color)
        t.write(label, align="center", font=("Consolas", 8, "normal"))

def draw_badge(t, x, y, text, bg_color="#1e293b", fg_color="#38bdf8", size=9):
    t.penup()
    t.goto(x, y)
    t.color(fg_color)
    t.write(text, font=("Consolas", size, "bold"))

def draw_bullet(t, x, y, text, color="#94a3b8", size=9):
    t.penup()
    t.goto(x, y)
    t.color("#00ffcc")
    t.dot(4)
    t.goto(x + 10, y - 4)
    t.color(color)
    t.write(text, font=("Consolas", size, "normal"))

def main():
    screen = setup_canvas()
    t = make_turtle()
    screen.tracer(1, 10) # Zapneme viditelné kreslení s mírnou animací pro potěchu oka

    # --- 1. ZÁHLAVÍ / HLAVIČKA ---
    t.pensize(3)
    t.goto(-610, 380)
    t.color("#00f0ff")
    t.write("CYBER-NEURON: ARCHITEKTURA, L1 TRANSIT PIPELINE & BEZPEČNOSTNÍ PILÍŘE", font=("Consolas", 15, "bold"))
    t.pensize(1)
    t.goto(-610, 360)
    t.color("#64748b")
    t.write("Projekt: pokec/test  |  Engine: BCE Gradient (0-alloc takt ~2.3 ms)  |  Standard: SCADA/ISO Audit Trail", font=("Consolas", 9, "normal"))

    t.goto(-610, 350)
    t.pendown()
    t.color("#1e293b")
    t.forward(1220)
    t.penup()

    # --- 2. VELKÉ OBLASTI / KONTEJNERY ---
    # Oblast A: Řadič / Orchestrátor (Vlevo a Střed nahoře)
    draw_box(t, -610, 330, 430, 480, "#131a26", "#3b82f6", "[ 1. ORCHESTRÁTOR (chain_runner.py) ]")
    
    # Oblast B: Hardware Socket / RAM Buňky
    draw_box(t, -150, 330, 410, 480, "#111827", "#10b981", "[ 2. HARDWARE SOCKET / CORE (neuron_node.py) ]")
    
    # Oblast C: Disková hygiena & Audit (Vpravo)
    draw_box(t, 290, 330, 320, 480, "#161b22", "#f59e0b", "[ 3. DISKOVÁ HYGIENA & AUDIT TRAIL ]")

    # --- OBSAH BLOKU 1: ORCHESTRÁTOR ---
    draw_badge(t, -590, 280, "A) FÁZE A: Surová data -> První průchod", fg_color="#60a5fa")
    draw_bullet(t, -580, 260, "Vstupy: X_env (AND/OR/XOR), y_true", "#cbd5e1")
    draw_bullet(t, -580, 240, "Konfigurace: max_cas_sekund=1.0s, batch=250", "#94a3b8")
    
    # L1 Transit Buffer Sub-Box
    draw_box(t, -590, 215, 390, 105, "#1e293b", "#06b6d4", "L1 TRANSIT BUFFER (Časový multiplexer)")
    draw_bullet(t, -575, 175, "Zero-copy sdílený buffer v paměti řadiče", "#38bdf8")
    draw_bullet(t, -575, 155, "Uloží aktivační výstupy Fáze A (pred_A)", "#e2e8f0")
    draw_bullet(t, -575, 135, "Vytváří matici X_stageB pro další buňku", "#94a3b8")

    draw_badge(t, -590, 85, "B) RESET SBĚRNICE & FÁZE B", fg_color="#60a5fa")
    draw_bullet(t, -580, 65, "Jádro NeuronCore ZŮSTÁVÁ V RAM (zero realloc)", "#a7f3d0")
    draw_bullet(t, -580, 45, "core.reset_weights() vyčistí stav vahami", "#94a3b8")
    draw_bullet(t, -580, 25, "Fáze B konzumuje data z L1 Bufferu", "#cbd5e1")
    draw_bullet(t, -580, 5, "Celková orchestrace obou fází: ~19.8 ms!", "#4ade80")

    # Extraktor & Validace v orchestrátoru
    draw_box(t, -590, -40, 390, 85, "#0f172a", "#818cf8", "EXTRAKTOR & VERIFY (Stage Guard)")
    draw_bullet(t, -575, -75, "verify_stage(meta, X_stage, y): round-trip kontrola", "#cbd5e1")
    draw_bullet(t, -575, -95, "Přísná shoda predikce proti nahrannému stavu", "#f43f5e")

    # --- OBSAH BLOKU 2: HARDWARE SOCKET (NeuronCore) ---
    draw_badge(t, -130, 280, "VÝPOČETNÍ JÁDRO (NeuronCore)", fg_color="#34d399")
    draw_bullet(t, -120, 260, "Vstupní vektor X: matice [N x M]", "#cbd5e1")
    draw_bullet(t, -120, 240, "Váhy W + Bias b (v jednom vektoru)", "#cbd5e1")
    
    # Matematické srdce Sub-Box
    draw_box(t, -130, 215, 370, 115, "#064e3b", "#10b981", "BCE ENGINE & DVOJITÁ ZÁVORA")
    draw_bullet(t, -115, 180, "Z = X * W + b  -->  A = Sigmoid(Z)", "#a7f3d0")
    draw_bullet(t, -115, 160, "Grad: dW = X.T * (A - Y) / N  (BCE bypass)", "#ecfdf5")
    draw_bullet(t, -115, 140, "Batch clock (250 epoch) -> OS čas nebrzdí", "#fef08a")
    draw_bullet(t, -115, 120, "STOP: target_loss (<0.01) NEBO max_time", "#f87171")

    # SCADA DASHBOARD
    draw_box(t, -130, 75, 370, 120, "#1e1e2e", "#c084fc", "SCADA TERMINÁL DASHBOARD (Pilíř 1)")
    draw_bullet(t, -115, 40, "Přísně zarovnané sloupce (Glanceability):", "#e9d5ff")
    draw_bullet(t, -115, 20, "| TAKT | STATUS | DŮVOD | LATENCE | EPOCHY | LOSS |", "#d8b4fe")
    draw_bullet(t, -115, 0, "Bleskový přehled operátora bez čtení logů", "#94a3b8")
    draw_bullet(t, -115, -20, "Podpora silent=True pro rychlopalbu / burst", "#94a3b8")

    # BEZPEČNOSTNÍ ZÁMKY
    draw_box(t, -130, -70, 370, 55, "#3b0764", "#e879f9", "ADMIN OVERRIDE BEZPEČNOSTNÍ BRÁNA")
    draw_bullet(t, -115, -100, "Výměna vah za chodu pouze s tokenem (ISO)", "#f5d0fe")

    # --- OBSAH BLOKU 3: DISK, WORM & AUDIT TRAIL ---
    draw_badge(t, 310, 280, "PILÍŘ 3: DISKOVÁ HYGIENA (dumps/)", fg_color="#fbbf24")
    
    draw_box(t, 310, 255, 280, 85, "#27272a", "#f59e0b", "WORM OBČANKA (_meta.json)")
    draw_bullet(t, 325, 220, "Read-Only zámek na disku (S_IREAD)", "#fef3c7")
    draw_bullet(t, 325, 200, "SHA-256 otisk vah, run_id, rodokmen", "#fef3c7")
    draw_bullet(t, 325, 180, "Atomický zápis (.tmp -> os.replace)", "#94a3b8")

    draw_box(t, 310, 150, 280, 75, "#27272a", "#38bdf8", "ČITELNÝ MOZEK (_weights.dat)")
    draw_bullet(t, 325, 115, "Čistý text: W1, W2, ... na řádcích", "#e0f2fe")
    draw_bullet(t, 325, 95, "Poslední řádek = Bias (žádný pickle!)", "#e0f2fe")

    draw_badge(t, 310, 50, "PILÍŘ 2: AUDIT TRAIL (logs/)", fg_color="#f87171")
    draw_box(t, 310, 30, 280, 155, "#3f1a1a", "#ef4444", "APPEND-ONLY HASH-CHAIN")
    draw_bullet(t, 325, -5, "node_events.jsonl", "#fee2e2")
    draw_bullet(t, 325, -25, "Každý záznam nese prev_event_hash", "#fca5a5")
    draw_bullet(t, 325, -45, "Kryptografický blockchain řetěz", "#fca5a5")
    draw_bullet(t, 325, -65, "Loguje TICK_OK i odražené útoky:", "#cbd5e1")
    draw_bullet(t, 325, -85, "SECURITY_REJECT & ADMIN_OVERRIDE", "#fecaca")
    draw_bullet(t, 325, -105, "Torture Suite: 23/23 testů v zelené!", "#4ade80")

    # --- DYNAMICKÉ SPOJOVACÍ ŠIPKY (DATAPATH) ---
    t.pensize(2)
    # 1. Řadič -> Hardware Socket (Injektování dat a spuštění taktu)
    draw_arrow(t, -180, 240, -150, 240, color="#38bdf8", label="1. Takt / Run (Data X, y)")

    # 2. Hardware Socket výstup -> L1 Transit Buffer
    draw_arrow(t, -150, 160, -200, 160, color="#06b6d4", label="2. pred_A -> L1 Buffer")

    # 3. L1 Transit Buffer -> Resetované jádro pro Fázi B
    draw_arrow(t, -200, 135, -150, 135, color="#a855f7", label="3. X_stageB (Fáze B)")

    # 4. Hardware Socket -> Diskový Extraktor (WORM zápis)
    draw_arrow(t, 260, 210, 290, 210, color="#f59e0b", label="4. Atomický WORM dump")

    # 5. Hardware Socket -> Append-only Audit deník
    draw_arrow(t, 260, -40, 290, -40, color="#ef4444", label="5. Hash-chain událost")

    # --- LEGENDA DOLE ---
    draw_box(t, -610, -170, 1220, 140, "#0b0f17", "#334155", "[ SYSTÉMOVÁ LEGENDA & PRINCIPY EVOLUCE ]")
    
    col1_x = -590
    draw_bullet(t, col1_x, -210, "Časový multiplexing: 1 fyzický socket v RAM obslouží celou pipeline bez memory-leaku a GC spikes.", "#cbd5e1", size=8)
    draw_bullet(t, col1_x, -230, "L1 Transit Buffer: Eliminuje diskový mezistupínek pro přenos signálu mezi vrstvami buněk.", "#cbd5e1", size=8)
    draw_bullet(t, col1_x, -250, "BCE Gradient Trick: Zrušení derivace sigmoidy zrychlilo učení ze 180 000 epoch na 500-1000.", "#cbd5e1", size=8)
    draw_bullet(t, col1_x, -270, "Preemptibilita: max_cas_sekund tvrdě garantuje splnění real-time limitu řídicího systému.", "#cbd5e1", size=8)

    col2_x = 20
    draw_bullet(t, col2_x, -210, "WORM (Write Once, Read Many): OS úroveň zámku (chmod S_IREAD) chrání občanku před manipulací.", "#cbd5e1", size=8)
    draw_bullet(t, col2_x, -230, "Auditní Hash-Chain: Žádná událost (ani sabotáž) nemůže být smazána bez porušení hash integrity.", "#cbd5e1", size=8)
    draw_bullet(t, col2_x, -250, "Torture Suite: 23 standalone útočných scénářů garantuje stabilitu při přechodu na nelinearity.", "#cbd5e1", size=8)
    draw_bullet(t, col2_x, -270, "Další krok: Paralelní buňky ve Fázi A (OR + NAND) -> L1 Buffer -> Fáze B (XOR syntéza).", "#38bdf8", size=8)

    # Finální refresh obrazovky
    screen.update()
    
    # Podpis a instrukce
    t.goto(0, -325)
    t.color("#64748b")
    t.write("Kliknutím do okna nebo stisknutím klávesy schéma zavřete.", align="center", font=("Consolas", 10, "italic"))
    screen.exitonclick()

if __name__ == "__main__":
    main()
