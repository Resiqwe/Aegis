"""
cpu_telemetry_probe.py - Hloubková hardwarová sonda a logický analyzátor pro Softwarové CPU.
Odposlouchává napětí, potenciály na sběrnici, vnitřní stav neuronových hradel a registrů v reálném čase.
"""

import os
import sys
import time
import numpy as np

from neuron import neuron_node as nn

class HardwareLogicProbe:
    """
    Logický analyzátor (Osciloskop & Telemetrická sonda).
    Odposlouchává:
      - Vstupní signály na sběrnici (A, B)
      - Vnitřní analogové napětí neuronu Z = (X * W + b) před aktivací
      - Pravděpodobnost / nasycení tranzistoru A = Sigmoid(Z)
      - Klopný práh (0 nebo 1)
      - Stav registrů a paměťových latchů
    """
    def __init__(self):
        self.channel_events = []

    def probe_gate(self, component: str, gate_name: str, inputs: list[float], W: np.ndarray, b: float, out_prob: float, out_bit: int):
        # Analogový potenciál na bázi tranzistoru/neuronu
        Z = float(np.dot(inputs, W) + b)
        event = {
            "component": component,
            "gate": gate_name,
            "inputs": inputs,
            "weights": W.tolist(),
            "bias": float(b),
            "analog_voltage_Z": Z,
            "saturation_sigmoid": float(out_prob),
            "digital_out": int(out_bit)
        }
        self.channel_events.append(event)
        return event

    def dump_oscilloscope_view(self):
        """Vytiskne detailní SCADA osciloskopickou stopu signálů."""
        print("\n" + "=" * 115)
        print("  HLOUBKOVÝ TELEMETRICKÝ ODPOSLECH SBĚRNICE & NEURONOVÝCH PŘECHODŮ (LOGIC ANALYZER)")
        print("=" * 115)
        header = f"| {'KOMPONENTA':<14} | {'HRADLO':<6} | {'VSTUP':<8} | {'POTENCIAL Z (V)':<16} | {'NASYCENI Sigm(Z)':<17} | {'LOGICKY BIT':<12} | {'STAV TRANZISTORU':<16} |"
        print(header)
        print("-" * 115)

        for ev in self.channel_events:
            inp_str = f"[{int(ev['inputs'][0])},{int(ev['inputs'][1])}]"
            z_val = f"{ev['analog_voltage_Z']:+8.3f} V"
            sat = f"{ev['saturation_sigmoid'] * 100:6.2f} %"
            bit_str = f"PIN = {ev['digital_out']}"
            
            # Stav sepnutí (hardwarová analogie tranzistoru)
            if ev['saturation_sigmoid'] > 0.95:
                state = "PLNE SEPNUTO (1)"
            elif ev['saturation_sigmoid'] < 0.05:
                state = "UZAVRENO (0)"
            else:
                state = "PRECHODOVY JEV"

            row = f"| {ev['component']:<14} | {ev['gate']:<6} | {inp_str:<8} | {z_val:<16} | {sat:<17} | {bit_str:<12} | {state:<16} |"
            print(row)

        print("=" * 115 + "\n")


class ProbedNeuronALU:
    """ALU jednotka s integrovanou telemetrickou sondou."""
    def __init__(self, probe: HardwareLogicProbe):
        self.probe = probe
        self.socket = nn.NeuronCore(input_dim=2, learning_rate=1.5, seed=42)
        self.microcode_rom = {}
        self._calibrate()

    def _calibrate(self):
        X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float64)
        targets = {
            "AND":  np.array([0, 0, 0, 1]),
            "OR":   np.array([0, 1, 1, 1]),
            "NAND": np.array([1, 1, 1, 0])
        }
        for gate, y in targets.items():
            self.socket.W = np.random.randn(2)
            self.socket.b = 0.0
            self.socket.train_fast(X, y, max_cas_sekund=0.5, target_loss=1e-3, batch_epochs=100)
            self.microcode_rom[gate] = (self.socket.W.copy(), float(self.socket.b))

    def eval_probed(self, component: str, gate_name: str, a: int, b: int) -> int:
        W, bias = self.microcode_rom[gate_name]
        self.socket.W = W
        self.socket.b = bias
        vec = np.array([[float(a), float(b)]])
        probs, dec = self.socket.predict(vec)
        
        # Telemetrický odposlech
        self.probe.probe_gate(
            component=component,
            gate_name=gate_name,
            inputs=[float(a), float(b)],
            W=W,
            b=bias,
            out_prob=probs[0],
            out_bit=dec[0]
        )
        return int(dec[0])

    def xor_probed(self, component: str, a: int, b: int) -> int:
        nand_out = self.eval_probed(f"{component}->L1", "NAND", a, b)
        or_out   = self.eval_probed(f"{component}->L1", "OR", a, b)
        return self.eval_probed(f"{component}->OUT", "AND", nand_out, or_out)

    def half_adder_probed(self, a: int, b: int) -> tuple[int, int]:
        sum_bit = self.xor_probed("HalfAdder_Sum", a, b)
        carry_bit = self.eval_probed("HalfAdder_Cry", "AND", a, b)
        return sum_bit, carry_bit


def run_telemetry_inspection():
    probe = HardwareLogicProbe()
    alu = ProbedNeuronALU(probe)

    print("\n[ SPUŠTĚNÍ ODPOSLECHU: Sčítání 1 + 1 v neuronové ALU ]")
    print("  Sledujeme tok elektronů a potenciálů skrz jednotlivá hradla...")
    
    # Provedeme sčítání 1 + 1 na Half-Adderu
    sum_bit, carry_bit = alu.half_adder_probed(1, 1)

    # Vykreslíme osciloskopický záznam
    probe.dump_oscilloscope_view()

    print(f"  VÝSLEDEK Z LOGICKÉHO ANALYZÁTORU:")
    print(f"    Součet (Sum bit)      = {sum_bit} (Očekáváno: 0)")
    print(f"    Přenos (Carry-out)   = {carry_bit} (Očekáváno: 1)")
    print(f"    Binárně: 1 + 1 = 0b{carry_bit}{sum_bit} (dekadicky: 2)")

if __name__ == "__main__":
    run_telemetry_inspection()
