# edo_questions_17_36.py
from __future__ import annotations
from typing import Dict
import matplotlib.pyplot as plt

from edo_core import (
    TrainingRegime, WeightsR, ODEParams, InjuryEvent, FatigueConfig,
    simulate_player, calibrate_params_from_ml
)

# ---------
# Utilidad de ploteo
# ---------
def plot_series(sim: Dict[str, list], title: str, show_FTM: bool = True, show_R: bool = True):
    t = sim["t"]
    if show_FTM:
        plt.figure()
        plt.plot(t, sim["F"], label="F")
        plt.plot(t, sim["T"], label="T")
        plt.plot(t, sim["M"], label="M")
        plt.title(title + " — F/T/M")
        plt.xlabel("Años")
        plt.ylabel("Score (0..100)")
        plt.legend()
        plt.grid(True)

    if show_R:
        plt.figure()
        plt.plot(t, sim["R"], label="R")
        plt.title(title + " — Rating R")
        plt.xlabel("Años")
        plt.ylabel("R")
        plt.legend()
        plt.grid(True)

    plt.show()

# ---------
# Defaults del proyecto (puedes ajustar)
# ---------
def base_setup(potential_pred: float = 85.0, position: str = "DEFAULT"):
    params, w = calibrate_params_from_ml(potential_pred=potential_pred, position=position)
    age0 = 22.0
    y0 = (65.0, 62.0, 60.0)  # F,T,M iniciales
    dt = 1/52  # semanal (años)
    return params, w, age0, y0, dt

# =========================
# Q17–Q36 (una función por pregunta)
# =========================

def q17():
    """EF=0.9, ET=0.3. Comportamiento de F y T en 5 años."""
    params, w, age0, y0, dt = base_setup()
    train = TrainingRegime(EF=0.9, ET=0.3, EM=0.6)
    sim = simulate_player(years=5, dt=dt, age0=age0, y0=y0, params=params, weights=w, train_regime=train)
    plot_series(sim, "Q17: EF=0.9, ET=0.3 (5 años)", show_R=False)
    return sim

def q18():
    """wF=0.7, wT=0.2, wM=0.1 y F decrece rápido => R cae fuerte."""
    params, _, age0, y0, dt = base_setup()
    w = WeightsR(0.7, 0.2, 0.1)
    # fuerza decadencia física: subimos betaF0
    params.betaF0 *= 1.8
    train = TrainingRegime(EF=0.4, ET=0.4, EM=0.4)
    sim = simulate_player(5, dt, age0, y0, params, w, train)
    plot_series(sim, "Q18: wF alto + F decrece rápido", show_FTM=True, show_R=True)
    return sim

def q19():
    """Lesión grave: reduce F 50% durante 6 meses."""
    params, w, age0, y0, dt = base_setup()
    injuries = [InjuryEvent(start_year=1.0, duration_years=0.5, severity=0.5, mode="shock")]
    train = TrainingRegime(0.7, 0.7, 0.6)
    sim = simulate_player(5, dt, age0, y0, params, w, train, injuries=injuries)
    plot_series(sim, "Q19: Lesión 50% por 6 meses")
    return sim

def q20():
    """Comparar joven promesa vs late bloomer en R."""
    params1, w1, age0, y0, dt = base_setup(potential_pred=92.0)
    params2, w2, _, _, _ = base_setup(potential_pred=82.0)

    # joven promesa: alta alpha temprano
    train_promesa = TrainingRegime(0.8, 0.8, 0.7)
    sim1 = simulate_player(10, dt, age0, y0, params1, w1, train_promesa)

    # late bloomer: intensidades moderadas al inicio y altas después
    train_early = TrainingRegime(0.5, 0.5, 0.5)
    sim2a = simulate_player(4, dt, age0, y0, params2, w2, train_early)
    y0b = (sim2a["F"][-1], sim2a["T"][-1], sim2a["M"][-1])
    train_late = TrainingRegime(0.85, 0.85, 0.7)
    sim2b = simulate_player(6, dt, age0+4, y0b, params2, w2, train_late)

    # plot solo R comparativo
    plt.figure()
    plt.plot(sim1["t"], sim1["R"], label="Joven promesa")
    t2 = sim2a["t"] + [x+4 for x in sim2b["t"]]
    R2 = sim2a["R"] + sim2b["R"]
    plt.plot(t2, R2, label="Late bloomer")
    plt.title("Q20: Comparación R — promesa vs late bloomer")
    plt.xlabel("Años")
    plt.ylabel("R")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {"promesa": sim1, "late": {"t": t2, "R": R2}}

def q21():
    """M cambia si delta es alto y F/T crecen rápido."""
    params, w, age0, y0, dt = base_setup()
    params.deltaM = 0.8  # delta alto
    train = TrainingRegime(0.9, 0.9, 0.4)
    sim = simulate_player(5, dt, age0, y0, params, w, train)
    plot_series(sim, "Q21: deltaM alto + F/T crecen rápido", show_FTM=True, show_R=False)
    return sim

def q22():
    """Aopt=28, sigma=2. ¿A qué edad alcanza pico físico?"""
    params, w, age0, y0, dt = base_setup()
    params.Aopt = 28.0
    params.sigma = 2.0
    train = TrainingRegime(0.7, 0.6, 0.6)
    sim = simulate_player(10, dt, age0, y0, params, w, train)
    plot_series(sim, "Q22: Aopt=28, sigma=2 (pico físico cerca de 28)", show_R=False)
    return sim

def q23():
    """Aumentar betaF => más decadencia física después de 30."""
    params, w, age0, y0, dt = base_setup()
    params.betaF0 *= 1.7
    train = TrainingRegime(0.6, 0.6, 0.6)
    sim = simulate_player(15, dt, age0, y0, params, w, train)
    plot_series(sim, "Q23: betaF aumentado (más decadencia post-30)", show_R=False)
    return sim

def q24():
    """Entrenamiento balanceado EF=ET=EM=0.7 por 10 años."""
    params, w, age0, y0, dt = base_setup()
    train = TrainingRegime(0.7, 0.7, 0.7)
    sim = simulate_player(10, dt, age0, y0, params, w, train)
    plot_series(sim, "Q24: Balanceado 10 años")
    return sim

def q25():
    """Fatiga acumulada afecta F y T."""
    params, w, age0, y0, dt = base_setup()
    train = TrainingRegime(0.8, 0.8, 0.6)
    fatigue = FatigueConfig(enabled=True, k=0.25, recovery=0.10, cap=1.0)
    sim = simulate_player(8, dt, age0, y0, params, w, train, fatigue_cfg=fatigue)
    plot_series(sim, "Q25: Fatiga acumulada", show_R=True)
    return sim

def q26():
    """Si A=34, ¿cuál es betaT? (según función beta_age del proyecto)."""
    from edo_core import beta_age
    params, _, _, _, _ = base_setup()
    age = 34.0
    betaT = beta_age(age, params.betaT0, pivot=30.0, slope=0.08)
    print(f"Q26: A=34 => betaT = {betaT:.4f} (con betaT0={params.betaT0:.4f})")
    return betaT

def q27():
    """alphaF bajo pero gammaFT alto. ¿Qué atributos se benefician?"""
    params, w, age0, y0, dt = base_setup()
    params.alphaF *= 0.4
    params.gammaFT *= 3.0
    train = TrainingRegime(0.6, 0.7, 0.6)
    sim = simulate_player(8, dt, age0, y0, params, w, train)
    plot_series(sim, "Q27: alphaF bajo + gammaFT alto (sinergia F<->T)")
    return sim

def q28():
    """¿Qué pasa si wF,wT,wM no suman 1?"""
    params, _, age0, y0, dt = base_setup()
    w_bad = WeightsR(0.7, 0.2, 0.2)  # suma 1.1
    train = TrainingRegime(0.7, 0.7, 0.7)

    sim_norm = simulate_player(5, dt, age0, y0, params, w_bad, train, normalize_weights_for_R=True)
    sim_raw  = simulate_player(5, dt, age0, y0, params, w_bad, train, normalize_weights_for_R=False)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(sim_norm["t"], sim_norm["R"], label="R con pesos normalizados")
    plt.plot(sim_raw["t"], sim_raw["R"], label="R sin normalizar (mal escalado)")
    plt.title("Q28: Pesos que no suman 1 => escala incorrecta")
    plt.xlabel("Años"); plt.ylabel("R"); plt.legend(); plt.grid(True)
    plt.show()
    return {"normalized": sim_norm, "raw": sim_raw}

def q29():
    """Lesión temporal con recuperación exponencial."""
    params, w, age0, y0, dt = base_setup()
    injuries = [InjuryEvent(start_year=2.0, duration_years=1.0, severity=0.5, mode="exp_recovery")]
    train = TrainingRegime(0.7, 0.7, 0.6)
    sim = simulate_player(6, dt, age0, y0, params, w, train, injuries=injuries)
    plot_series(sim, "Q29: Lesión con recuperación exponencial")
    return sim

def q30():
    """Solo técnica: ET=1, EF=EM=0. R depende de pesos."""
    params, w, age0, y0, dt = base_setup()
    train = TrainingRegime(EF=0.0, ET=1.0, EM=0.0)
    sim = simulate_player(6, dt, age0, y0, params, w, train)
    plot_series(sim, "Q30: Solo técnica (ET=1)", show_FTM=True, show_R=True)
    return sim

def q31():
    """Pico técnico a los 32 y luego decae."""
    params, w, age0, y0, dt = base_setup()
    # forzamos que betaT suba más después de 32
    params.betaT0 *= 1.0
    train = TrainingRegime(0.5, 0.9, 0.6)
    sim = simulate_player(15, dt, age0, y0, params, w, train)
    plot_series(sim, "Q31: Pico técnico cercano a 32 y luego decae", show_R=True)
    return sim

def q32():
    """Si sigma es muy pequeño en el término gaussiano."""
    params, w, age0, y0, dt = base_setup()
    params.sigma = 0.4  # muy pequeño => pico muy estrecho
    train = TrainingRegime(0.7, 0.6, 0.6)
    sim = simulate_player(12, dt, age0, y0, params, w, train)
    plot_series(sim, "Q32: sigma muy pequeño (pico estrecho)", show_R=False)
    return sim

def q33():
    """Evolución de un portero con pesos distintos en R."""
    params, w, age0, y0, dt = base_setup(position="GK")
    train = TrainingRegime(0.5, 0.8, 0.7)
    sim = simulate_player(12, dt, age0, y0, params, w, train)
    plot_series(sim, "Q33: Portero (pesos y Aopt distintos)")
    return sim

def q34():
    """Trayectoria con Aopt=25 vs Aopt=30."""
    params1, w, age0, y0, dt = base_setup()
    params2, w2, _, _, _ = base_setup()
    params1.Aopt = 25.0
    params2.Aopt = 30.0
    train = TrainingRegime(0.7, 0.7, 0.6)

    s1 = simulate_player(12, dt, age0, y0, params1, w, train)
    s2 = simulate_player(12, dt, age0, y0, params2, w2, train)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(s1["t"], s1["F"], label="F (Aopt=25)")
    plt.plot(s2["t"], s2["F"], label="F (Aopt=30)")
    plt.title("Q34: Aopt 25 vs 30 (F)")
    plt.xlabel("Años"); plt.ylabel("F"); plt.legend(); plt.grid(True)
    plt.show()
    return {"Aopt25": s1, "Aopt30": s2}

def q35():
    """Se lesiona cada 2 años. Efecto en rating máximo."""
    params, w, age0, y0, dt = base_setup()
    injuries = []
    # cada 2 años: lesión 3 meses con severidad 35%
    t = 1.0
    while t < 12.0:
        injuries.append(InjuryEvent(start_year=t, duration_years=0.25, severity=0.35, mode="shock"))
        t += 2.0

    train = TrainingRegime(0.75, 0.75, 0.6)
    sim = simulate_player(12, dt, age0, y0, params, w, train, injuries=injuries)
    plot_series(sim, "Q35: Lesión cada 2 años")
    print(f"Q35: R_max = {max(sim['R']):.2f}")
    return sim

def q36():
    """¿Qué régimen de entrenamiento maximiza R a los 28? (búsqueda simple)."""
    params, w, age0, y0, dt = base_setup()
    target_age = 28.0
    years = max(0.0, target_age - age0)

    best = None
    best_cfg = None

    # grid search simple sobre EF,ET,EM (0.4..0.9)
    candidates = [0.4, 0.55, 0.7, 0.85, 0.9]
    for EF in candidates:
        for ET in candidates:
            for EM in candidates:
                train = TrainingRegime(EF, ET, EM)
                sim = simulate_player(years, dt, age0, y0, params, w, train)
                R_at = sim["R"][-1]
                if (best is None) or (R_at > best):
                    best = R_at
                    best_cfg = (EF, ET, EM)

    print(f"Q36: Mejor régimen para maximizar R a los {target_age} años:")
    print(f"     EF={best_cfg[0]}, ET={best_cfg[1]}, EM={best_cfg[2]}  => R={best:.2f}")

    # mostrar la trayectoria del mejor
    train_best = TrainingRegime(*best_cfg)
    sim_best = simulate_player(years, dt, age0, y0, params, w, train_best)
    plot_series(sim_best, f"Q36: Mejor régimen (R@{target_age} máximo)")
    return {"best_cfg": best_cfg, "R_at_28": best, "sim": sim_best}
