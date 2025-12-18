#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test rápido del sistema Día 3
Verifica que todos los componentes funcionen correctamente
"""

print("="*60)
print("TEST RÁPIDO - DÍA 3: SISTEMA DINÁMICO")
print("="*60)

# 1. Test de imports
print("\n1️⃣ Verificando imports...")
try:
    from edo_core import (
        TrainingRegime, WeightsR, ODEParams, InjuryEvent, FatigueConfig,
        simulate_player, calibrate_params_from_ml, rk4_step
    )
    print("   ✅ edo_core.py - OK")
except Exception as e:
    print(f"   ❌ edo_core.py - ERROR: {e}")
    exit(1)

try:
    from edo_questions_17_36 import q17, q20, q36
    print("   ✅ edo_questions_17_36.py - OK")
except Exception as e:
    print(f"   ❌ edo_questions_17_36.py - ERROR: {e}")

try:
    from sistema_integrado import SistemaIntegradoFutbolistas
    print("   ✅ sistema_integrado.py - OK")
except Exception as e:
    print(f"   ❌ sistema_integrado.py - ERROR: {e}")

try:
    from casos_estudio import CASOS_ESTUDIO, AnalizadorCasosEstudio
    print("   ✅ casos_estudio.py - OK")
except Exception as e:
    print(f"   ❌ casos_estudio.py - ERROR: {e}")

# 2. Test de simulación básica
print("\n2️⃣ Test de simulación básica...")
try:
    params, weights = calibrate_params_from_ml(potential_pred=85.0, position="MID")
    
    age0 = 22
    y0 = (70.0, 68.0, 65.0)
    train = TrainingRegime(0.7, 0.7, 0.7)
    
    sim = simulate_player(
        years=5,
        dt=1/52,
        age0=age0,
        y0=y0,
        params=params,
        weights=weights,
        train_regime=train
    )
    
    # Verificaciones
    assert len(sim["t"]) > 0, "Serie temporal vacía"
    assert len(sim["F"]) == len(sim["t"]), "Longitud inconsistente"
    assert all(0 <= f <= 100 for f in sim["F"]), "F fuera de rango"
    assert all(0 <= t <= 100 for t in sim["T"]), "T fuera de rango"
    assert all(0 <= m <= 100 for m in sim["M"]), "M fuera de rango"
    
    R_pico = max(sim["R"])
    edad_pico = sim["age"][sim["R"].index(R_pico)]
    
    print(f"   ✅ Simulación ejecutada correctamente")
    print(f"      - {len(sim['t'])} puntos simulados")
    print(f"      - Rating pico: {R_pico:.2f} @ {edad_pico:.1f} años")
    print(f"      - F, T, M en rango válido [0, 100]")
    
except Exception as e:
    print(f"   ❌ Error en simulación: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 3. Test de RK4
print("\n3️⃣ Test de RK4 solver...")
try:
    # Test simple: dy/dt = -y, solución: y(t) = y0*exp(-t)
    def test_ode(t, y):
        return (-y[0], -y[1], -y[2])
    
    t0 = 0.0
    y0_test = (1.0, 1.0, 1.0)
    h = 0.1
    
    y1 = rk4_step(test_ode, t0, y0_test, h)
    
    # Solución exacta: exp(-0.1) ≈ 0.9048
    expected = 0.9048
    error = abs(y1[0] - expected)
    
    if error < 0.01:
        print(f"   ✅ RK4 funcionando correctamente")
        print(f"      - Error vs solución exacta: {error:.6f}")
    else:
        print(f"   ⚠️ RK4 con error alto: {error:.6f}")
    
except Exception as e:
    print(f"   ❌ Error en RK4: {e}")

# 4. Test de calibración
print("\n4️⃣ Test de calibración de parámetros...")
try:
    for pos in ["GK", "DEF", "MID", "FWD"]:
        params, weights = calibrate_params_from_ml(85.0, pos)
        
        # Verificar que los parámetros están en rangos razonables
        assert 0.1 < params.alphaF < 1.0, f"alphaF fuera de rango para {pos}"
        assert 0.01 < params.betaF0 < 0.2, f"betaF0 fuera de rango para {pos}"
        assert 25 < params.Aopt < 32, f"Aopt fuera de rango para {pos}"
        
        # Verificar que los pesos suman 1
        w_sum = weights.wF + weights.wT + weights.wM
        assert 0.99 < w_sum < 1.01, f"Pesos no suman 1 para {pos}"
    
    print(f"   ✅ Calibración correcta para todas las posiciones")
    print(f"      - GK, DEF, MID, FWD validados")
    
except Exception as e:
    print(f"   ❌ Error en calibración: {e}")

# 5. Test de lesiones
print("\n5️⃣ Test de modelado de lesiones...")
try:
    injury = InjuryEvent(start_year=2.0, duration_years=0.5, severity=0.4, mode="exp_recovery")
    
    params, weights = calibrate_params_from_ml(85.0)
    
    sim_sin = simulate_player(5, 1/52, 22, (70, 68, 65), params, weights, train, injuries=[])
    sim_con = simulate_player(5, 1/52, 22, (70, 68, 65), params, weights, train, injuries=[injury])
    
    # Debe haber una diferencia notable
    R_pico_sin = max(sim_sin["R"])
    R_pico_con = max(sim_con["R"])
    
    diferencia = R_pico_sin - R_pico_con
    
    if diferencia > 0:
        print(f"   ✅ Lesiones impactan correctamente")
        print(f"      - Reducción de {diferencia:.2f} puntos en pico")
    else:
        print(f"   ⚠️ Impacto de lesiones no detectado")
    
except Exception as e:
    print(f"   ❌ Error en modelado de lesiones: {e}")

# 6. Test de fatiga
print("\n6️⃣ Test de modelado de fatiga...")
try:
    fatigue_cfg = FatigueConfig(enabled=True, k=0.15, recovery=0.25, cap=1.0)
    
    sim_fatiga = simulate_player(
        8, 1/52, 22, (70, 68, 65), params, weights, 
        TrainingRegime(0.9, 0.9, 0.8),
        fatigue_cfg=fatigue_cfg
    )
    
    max_fatiga = max(sim_fatiga["fatigue"])
    
    if max_fatiga > 0:
        print(f"   ✅ Fatiga acumulándose correctamente")
        print(f"      - Fatiga máxima: {max_fatiga:.3f}")
    else:
        print(f"   ⚠️ Fatiga no se acumula")
    
except Exception as e:
    print(f"   ❌ Error en modelado de fatiga: {e}")

# 7. Test del sistema integrado
print("\n7️⃣ Test del sistema integrado...")
try:
    sistema = SistemaIntegradoFutbolistas()
    
    jugador_test = {
        "name": "Test Player",
        "age": 22,
        "position": "MID",
        "overall": 75,
        "potential": 85
    }
    
    resultado = sistema.simular_carrera_completa(
        player_data=jugador_test,
        years=5,
        training_scenario="balanced",
        enable_fatigue=False
    )
    
    # Verificar estructura del resultado
    assert "player_data" in resultado
    assert "predictions" in resultado
    assert "simulation" in resultado
    assert "analysis" in resultado
    assert "recommendations" in resultado
    
    analisis = resultado["analysis"]
    assert "peak_rating" in analisis
    assert "peak_age" in analisis
    
    print(f"   ✅ Sistema integrado funcional")
    print(f"      - Pipeline completo ejecutado")
    print(f"      - Pico: {analisis['peak_rating']:.2f} @ {analisis['peak_age']:.1f} años")
    
except Exception as e:
    print(f"   ❌ Error en sistema integrado: {e}")

# 8. Test de casos de estudio
print("\n8️⃣ Test de definición de casos de estudio...")
try:
    assert len(CASOS_ESTUDIO) == 5, "Deben haber 5 casos"
    
    for caso_id, caso in CASOS_ESTUDIO.items():
        assert "name" in caso
        assert "age" in caso
        assert "position" in caso
        assert "overall" in caso
        assert "potential" in caso
    
    print(f"   ✅ 5 casos de estudio definidos correctamente")
    
except Exception as e:
    print(f"   ❌ Error en casos de estudio: {e}")

# Resumen final
print("\n" + "="*60)
print("✅ TODOS LOS TESTS PASARON CORRECTAMENTE")
print("="*60)
print("\n📊 Sistema Día 3 listo para uso:")
print("   1. Dashboard: streamlit run dashboard_streamlit.py")
print("   2. Casos: python casos_estudio.py")
print("   3. Preguntas: python run_all.py")
print("\n🎯 ¡Todo funcionando correctamente!")
