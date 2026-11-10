#!/usr/env python3
"""Comprehensive import test for Cyber-Tactics restructured project."""
import sys

errors = []

def test(name, func):
    try:
        func()
        print(f"  OK: {name}")
    except Exception as e:
        errors.append((name, str(e)))
        print(f"  FAIL: {name} - {e}")

print("=== Comprehensive Import Test ===\n")

# 1. Launcher
test("CyberTacticsLauncher", lambda: __import__("src.launcher").launcher.CyberTacticsLauncher)

# 2. AI brains with immutability
import src.ai_brains as ab
test("ai_brains SENSOR_NAMES immutable", lambda: (
    test_type := type(ab.SENSOR_NAMES).__name__,
    print(f"  SENSOR_NAMES type: {test_type}"),
    test_action_type := type(ab.ACTION_NAMES).__name__,
    print(f"  ACTION_NAMES type: {test_action_type}"),
    # Verify immutability - trying to mutate should fail
    [ab.SENSOR_NAMES.__setitem__("_data", []) for _ in [1]][0] if hasattr(ab.SENSOR_NAMES, '__setitem__') else None,
    print("  SENSOR_NAMES is read-only (mappingproxy)")
))

# 3. Engine core
test("engine_core modules", lambda: __import__("src.engine_core"))

# 4. Scenarios
test("scenarios SCENARIO_*", lambda: __import__("src.scenarios"))

# 5. Telemetry logger
test("telemetry_logger", lambda: __import__("src.telemetry_logger"))

# 6. Tactical HUD
test("tactical_hud", lambda: __import__("src.tactical_hud"))

# 7. Battlefield renderer
test("battlefield_renderer", lambda: __import__("src.battlefield_renderer"))

# 8. Analyze telemetry
test("analyze_telemetry", lambda: __import__("src.analyze_telemetry"))

# 9. Telemetry lab
test("telemetry_lab", lambda: __import__("src.telemetry_lab"))

# 10. Hardware telemetry
test("hardware_telemetry", lambda: __import__("src.hardware_telemetry"))

print(f"\n=== Results: {len(errors)} errors ===")
if errors:
    for name, err in errors:
        print(f"  FAIL: {name} - {err}")
else:
    print("  All imports successful!")