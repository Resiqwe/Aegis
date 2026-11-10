"""
turtle_run_flow.py - Animovaný průběh jednoho TAKTU (RUN) našeho kyber-neuronu.
Ukazuje detailní krok-za-krokem flow od příchodu dat až po auditní pečeť.
"""

import turtle
import time

def setup_canvas():
    screen = turtle.Screen()
    screen.setup(width=1350, height=880)
    screen.bgcolor("#0a0f16")
    screen.title("ANATOMIE JEDNOHO TAKTU (RUN): Detailní časová osa a tok řízení")
    screen.tracer(0)
    return screen

def make_turtle(color="#00f0ff"):
    t = turtle.Turtle()
    t.hideturtle()
    t.speed(0)
    t.pensize(2)
    t.color(color)
    t.penup()
    return t

def draw_header(t):
    t.goto(-630, 390)
    t.color("#38bdf8")
    t.write("CYBER-NEURON: ANATOMIE A TOK JEDNOHO TAKTU (RUN PIPELINE)", font=("Consolas", 15, "bold"))
    t.goto(-630, 370)
    t.color("#64748b")
    t.write("Časový sled: Inicializace -> Dávkovaná BCE smyčka -> Validace -> WORM Pečeť -> SCADA výpis (~2.3 ms)", font=("Consolas", 9, "normal"))
    
    t.goto(-630, 360)
    t.pendown()
    t.color("#1e293b")
    t.forward(1260)
    t.penup()

def draw_step_card(t, x, y, w, h, step_num, title, subtitle, bullets, border_color="#38bdf8", fill_color="#111827", accent="#00ffcc"):
    # Pozadí a rámeček
    t.goto(x, y)
    t.setheading(0)
    t.color(border_color, fill_color)
    t.pendown()
    t.begin_fill()
    for _ in range(2):
        t.forward(w)
        t.right(90)
        t.forward(h)
        t.right(90)
    t.end_fill()
    t.penup()

    # Kolečko s číslem kroku
    t.goto(x + 22, y - 26)
    t.color(border_color)
    t.dot(26)
    t.goto(x + 22, y - 33)
    t.color("#0a0f16")
    t.write(str(step_num), align="center", font=("Consolas", 11, "bold"))

    # Nadpis karty
    t.goto(x + 45, y - 24)
    t.color("#f8fafc")
    t.write(title, font=("Consolas", 11, "bold"))

    t.goto(x + 45, y - 40)
    t.color(accent)
    t.write(subtitle, font=("Consolas", 8, "italic"))

    # Odrážky
    curr_y = y - 62
    for b in bullets:
        t.goto(x + 18, curr_y)
        t.color(accent)
        t.dot(3)
        t.goto(x + 28, curr_y - 4)
        t.color("#94a3b8")
        t.write(b, font=("Consolas", 8, "normal"))
        curr_y -= 18

def draw_flow_arrow(t, x1, y1, x2, y2, label="", color="#38bdf8"):
    t.penup()
    t.goto(x1, y1)
    t.setheading(t.towards(x2, y2))
    t.color(color)
    t.pendown()
    t.pensize(2)
    t.goto(x2, y2)
    t.stamp()
    t.penup()
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        t.goto(mx, my + 8)
        t.color(color)
        t.write(label, align="center", font=("Consolas", 8, "bold"))

def main():
    screen = setup_canvas()
    t = make_turtle()
    screen.tracer(1, 15)

    draw_header(t)

    # -------------------------------------------------------------
    # ŘADA 1: OD VSTUPU PO PŘÍPRAVU JÁDRA
    # -------------------------------------------------------------
    card_w = 285
    card_h = 165
    gap_x = 40
    y_row1 = 330

    # KROK 1: PŘÍCHOD DAT A INJEKCE
    draw_step_card(t, -630, y_row1, card_w, card_h, 
                   step_num=1, 
                   title="INJEKCE VSTUPU", 
                   subtitle="Volání core.run(X, y, max_time)", 
                   bullets=[
                       "Vstupní matice X: 4 řádky x 2 sloupce",
                       "Cílový vektor y: [0, 1, 1, 1] (např. OR)",
                       "Prevence realloc: data sedí v RAM",
                       "Nastavení watchdogu (max_cas=1.0s)"
                   ],
                   border_color="#3b82f6", fill_color="#131d2e", accent="#60a5fa")

    # Šipka 1 -> 2
    draw_flow_arrow(t, -630 + card_w, y_row1 - 80, -630 + card_w + gap_x, y_row1 - 80, label="Data in", color="#60a5fa")

    # KROK 2: STAV VAH A WARM-START CHECK
    draw_step_card(t, -630 + (card_w + gap_x), y_row1, card_w, card_h,
                   step_num=2,
                   title="INICIALIZACE VAH",
                   subtitle="Kontrola sanity & Rodokmen",
                   bullets=[
                       "Máme váhy? Warm-start (adaptace)",
                       "Nebo prázdno? np.random.uniform",
                       "Sanitární kontrola: detekce NaN/Inf",
                       "Zaznamenání počátečních vah do RAM"
                   ],
                   border_color="#06b6d4", fill_color="#0e2330", accent="#22d3ee")

    # Šipka 2 -> 3
    draw_flow_arrow(t, -630 + 2*(card_w + gap_x) - gap_x + gap_x, y_row1 - 80, -630 + 2*(card_w + gap_x), y_row1 - 80, label="Ready", color="#22d3ee")

    # KROK 3: MATEMATICKÉ SRDCE (BCE VÝPOČET)
    draw_step_card(t, -630 + 2*(card_w + gap_x), y_row1, card_w, card_h,
                   step_num=3,
                   title="FORWARD & GRADIENT",
                   subtitle="Z = X*W + b  -->  A = σ(Z)",
                   bullets=[
                       "Aktivace: A = 1 / (1 + exp(-Z))",
                       "Loss: BCE = -mean(y*log(A) + (1-y)...)",
                       "Gradient: dW = X.T * (A - y) / N",
                       "BCE bypass: žádné násobení A*(1-A)!"
                   ],
                   border_color="#10b981", fill_color="#062e24", accent="#34d399")

    # Šipka 3 -> 4
    draw_flow_arrow(t, -630 + 3*(card_w + gap_x) - gap_x + gap_x, y_row1 - 80, -630 + 3*(card_w + gap_x), y_row1 - 80, label="Epocha", color="#34d399")

    # KROK 4: HODINOVÁ SMYČKA & DÁVKOVÁNÍ
    draw_step_card(t, -630 + 3*(card_w + gap_x), y_row1, card_w, card_h,
                   step_num=4,
                   title="DÁVKOVANÉ HODINY",
                   subtitle="Epoch Batching (modulo 250)",
                   bullets=[
                       "249 epoch bez dotazu na hodiny OS",
                       "Každých 250 epoch: time.time() check",
                       "Podmínka 1: loss <= 0.01 -> TARGET_LOSS",
                       "Podmínka 2: elapsed >= limit -> TIMEOUT",
                       "Výsledek: nulová režie plánovače OS"
                   ],
                   border_color="#eab308", fill_color="#2b2308", accent="#facc15")

    # -------------------------------------------------------------
    # PŘECHODOVÁ ŠIPKA Z ŘADY 1 DO ŘADY 2 (Zprava dolů a doleva)
    # -------------------------------------------------------------
    t.penup()
    t.goto(-630 + 3*(card_w + gap_x) + card_w/2, y_row1 - card_h)
    t.pendown()
    t.color("#a855f7")
    t.pensize(2)
    t.setheading(270)
    t.forward(45)
    t.stamp()
    t.penup()
    t.goto(-630 + 3*(card_w + gap_x) + card_w/2 + 10, y_row1 - card_h - 25)
    t.color("#c084fc")
    t.write("Exit smyčky", font=("Consolas", 8, "bold"))

    # -------------------------------------------------------------
    # ŘADA 2: VÝSTUPNÍ HYGIENA, VALIDACE A PEČEŤ
    # -------------------------------------------------------------
    y_row2 = 100

    # KROK 5: SYNCHRONIZACE & EXTRAKCE
    draw_step_card(t, -630 + 3*(card_w + gap_x), y_row2, card_w, card_h,
                   step_num=5,
                   title="RE-SYNCHRONIZACE",
                   subtitle="Fix: Predikce vs. Poslední W",
                   bullets=[
                       "Z = X * W_final + b_final (čerstvý výpočet)",
                       "y_pred = (A >= 0.5).astype(int)",
                       "Prevence desynchronizace o 1 krok!",
                       "Příprava payloadu pro diskový dump"
                   ],
                   border_color="#a855f7", fill_color="#271538", accent="#c084fc")

    # Šipka 5 <- 6 (zprava doleva)
    draw_flow_arrow(t, -630 + 3*(card_w + gap_x), y_row2 - 80, -630 + 2*(card_w + gap_x) + card_w, y_row2 - 80, label="Payload", color="#c084fc")

    # KROK 6: WORM ZÁPIS & ATOMIC REPLACE
    draw_step_card(t, -630 + 2*(card_w + gap_x), y_row2, card_w, card_h,
                   step_num=6,
                   title="WORM DISKOVÝ DUMP",
                   subtitle="Atomický zápis .tmp -> replace",
                   bullets=[
                       "1. Zápis _weights.dat (čistý text: W, b)",
                       "2. SHA-256 otisk mozku vypočten",
                       "3. Zápis _meta.json s rodokmenem",
                       "4. os.chmod(S_IREAD): Zámek proti přepisu",
                       "5. Atomická aktualizace HEAD.json"
                   ],
                   border_color="#f97316", fill_color="#30190a", accent="#fb923c")

    # Šipka 6 <- 7
    draw_flow_arrow(t, -630 + 2*(card_w + gap_x), y_row2 - 80, -630 + (card_w + gap_x) + card_w, y_row2 - 80, label="Audit", color="#fb923c")

    # KROK 7: APPEND-ONLY HASH-CHAIN (LOG)
    draw_step_card(t, -630 + (card_w + gap_x), y_row2, card_w, card_h,
                   step_num=7,
                   title="HASH-CHAIN AUDIT",
                   subtitle="node_events.jsonl (Blockchain)",
                   bullets=[
                       "Načtení posledního prev_event_hash",
                       "Zabalení události: TICK_OK, Latence, Loss",
                       "Výpočet nového SHA-256 event_hash",
                       "Atomický append na konec souboru",
                       "Kryptografická nezvratnost historie"
                   ],
                   border_color="#ef4444", fill_color="#381010", accent="#f87171")

    # Šipka 7 <- 8
    draw_flow_arrow(t, -630 + (card_w + gap_x), y_row2 - 80, -630 + card_w, y_row2 - 80, label="Status", color="#f87171")

    # KROK 8: SCADA DASHBOARD & ZÁVĚR
    draw_step_card(t, -630, y_row2, card_w, card_h,
                   step_num=8,
                   title="SCADA VÝPIS & HOT",
                   subtitle="Okamžitá glanceability",
                   bullets=[
                       "Zarovnaný řádek: TAKT, DŮVOD, LATENCE",
                       "Pokud silent=True (burst): tichý režim",
                       "Návrat slovníku vysledky_behu volajícímu",
                       "Latence celého taktu: ~2.3 ms!",
                       "Socket připraven pro další takt v RAM"
                   ],
                   border_color="#10b981", fill_color="#062e24", accent="#34d399")

    # -------------------------------------------------------------
    # DETAILNÍ ČASOVÝ ROZPAD (LATENČNÍ BUDGET) DOLE
    # -------------------------------------------------------------
    draw_step_card(t, -630, -85, 1260, 215,
                   step_num="Σ",
                   title="LATENČNÍ ROZPAD A FYZIKÁLNÍ LIMIT JEDNOHO TAKTU (CELKEM: ~2.3 ms)",
                   subtitle="Kde se bere extrémní rychlost a proč je architektura preemptibilní",
                   bullets=[],
                   border_color="#475569", fill_color="#0c131d", accent="#38bdf8")

    # Sub-sloupce v rekapitulaci
    t.goto(-605, -150)
    t.color("#38bdf8")
    t.write("[ FÁZE 1: RAM & NUMPY ]", font=("Consolas", 10, "bold"))
    t.goto(-605, -170)
    t.color("#cbd5e1")
    t.write("• Čas: ~0.8 - 1.1 ms (500 epoch)", font=("Consolas", 8, "normal"))
    t.goto(-605, -190)
    t.color("#94a3b8")
    t.write("• Žádná alokace nových matic v RAM", font=("Consolas", 8, "normal"))
    t.goto(-605, -210)
    t.color("#94a3b8")
    t.write("• BCE gradient eliminuje 179 000 epoch", font=("Consolas", 8, "normal"))
    t.goto(-605, -230)
    t.color("#94a3b8")
    t.write("• Dávkované OS hodiny (2 dotazy/run)", font=("Consolas", 8, "normal"))

    t.goto(-180, -150)
    t.color("#fb923c")
    t.write("[ FÁZE 2: DISKOVÝ I/O & WORM ]", font=("Consolas", 10, "bold"))
    t.goto(-180, -170)
    t.color("#cbd5e1")
    t.write("• Čas: ~0.9 - 1.2 ms (SSD disk)", font=("Consolas", 8, "normal"))
    t.goto(-180, -190)
    t.color("#94a3b8")
    t.write("• Zápis malého _weights.dat (ASCII)", font=("Consolas", 8, "normal"))
    t.goto(-180, -210)
    t.color("#94a3b8")
    t.write("• SHA-256 hashování payloadu", font=("Consolas", 8, "normal"))
    t.goto(-180, -230)
    t.color("#94a3b8")
    t.write("• Zámek S_IREAD + HEAD atomic swap", font=("Consolas", 8, "normal"))

    t.goto(245, -150)
    t.color("#f87171")
    t.write("[ FÁZE 3: AUDIT CHAIN & DASH ]", font=("Consolas", 10, "bold"))
    t.goto(245, -170)
    t.color("#cbd5e1")
    t.write("• Čas: ~0.2 - 0.3 ms", font=("Consolas", 8, "normal"))
    t.goto(245, -190)
    t.color("#94a3b8")
    t.write("• Jediný atomický append do JSONL", font=("Consolas", 8, "normal"))
    t.goto(245, -210)
    t.color("#94a3b8")
    t.write("• Zpětné ověření hash-linku", font=("Consolas", 8, "normal"))
    t.goto(245, -230)
    t.color("#4ade80")
    t.write("• SCADA tisk na terminál konzole", font=("Consolas", 8, "bold"))

    screen.update()

    t.goto(0, -320)
    t.color("#64748b")
    t.write("Kliknutím do okna schéma zavřete.", align="center", font=("Consolas", 10, "italic"))
    screen.exitonclick()

if __name__ == "__main__":
    main()
