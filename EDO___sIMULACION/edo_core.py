# edo_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import math

# =========================
# Tipos y Config
# =========================

@dataclass
class TrainingRegime:
    """Intensidades de entrenamiento en [0,1]."""
    EF: float = 0.7  # físico
    ET: float = 0.7  # técnica
    EM: float = 0.7  # mental

@dataclass
class WeightsR:
    """Pesos para el Overall Rating R."""
    wF: float = 0.4
    wT: float = 0.4
    wM: float = 0.2

    def normalized(self) -> "WeightsR":
        s = self.wF + self.wT + self.wM
        if s == 0:
            return WeightsR(1/3, 1/3, 1/3)
        return WeightsR(self.wF/s, self.wT/s, self.wM/s)

@dataclass
class ODEParams:
    """Parámetros del sistema dinámico."""
    # crecimiento por entrenamiento
    alphaF: float = 0.35
    alphaT: float = 0.25
    alphaM: float = 0.20

    # decaimiento base (edad/fatiga)
    betaF0: float = 0.08
    betaT0: float = 0.06
    betaM0: float = 0.04

    # interacción / sinergias
    gammaFT: float = 0.03
    gammaFM: float = 0.02
    gammaTM: float = 0.02

    # edad óptima y anchura del pico físico (gauss)
    Aopt: float = 28.0
    sigma: float = 2.0

    # sensibilidad de M a cambios (tu “delta” conceptual)
    deltaM: float = 0.20

    # límites (escala 0..100)
    Fmax: float = 100.0
    Tmax: float = 100.0
    Mmax: float = 100.0

    # Slopes de envejecimiento (nuevo)
    slopeF: float = 0.10
    slopeT: float = 0.08
    slopeM: float = 0.06

@dataclass
class InjuryEvent:
    """Evento de lesión."""
    start_year: float
    duration_years: float
    severity: float  # 0.5 => reduce 50%
    mode: str = "shock"  # "shock" o "exp_recovery"

@dataclass
class FatigueConfig:
    """Fatiga acumulada por competiciones."""
    enabled: bool = False
    k: float = 0.12        # cuánto sube fatiga con el tiempo
    recovery: float = 0.25 # cuánto baja fatiga si entreno suave
    cap: float = 1.0       # fatiga máxima

# =========================
# Helpers
# =========================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def gaussian_peak(age: float, Aopt: float, sigma: float) -> float:
    # pico ~1 en Aopt, cae según sigma
    if sigma <= 1e-9:
        return 0.0
    z = (age - Aopt) / sigma
    return math.exp(-0.5 * z * z)

def compute_R(F: float, T: float, M: float, w: WeightsR, normalize_weights: bool = True) -> float:
    ww = w.normalized() if normalize_weights else w
    return ww.wF * F + ww.wT * T + ww.wM * M

# =========================
# Calibración (simple, usable)
# =========================
def calibrate_params_from_ml(
    potential_pred: float,
    position: str = "DEFAULT",
) -> Tuple[ODEParams, WeightsR]:
    """
    Usa 'potential_pred' (de tu MLP) para setear magnitudes de alpha/beta y Aopt por posición.
    Esta calibración es pragmática: mapea potential alto => mejor crecimiento y menor decaimiento.
    """
    pot = clamp(float(potential_pred), 40.0, 99.0)
    s = (pot - 40.0) / (99.0 - 40.0)  # 0..1

    # Aopt por posición (ajústalo a tu criterio/proyecto)
    pos_Aopt = {
        "GK": 30.0, "DEF": 28.5, "MID": 27.5, "FWD": 26.5,
        "DEFAULT": 28.0
    }
    Aopt = pos_Aopt.get(position.upper(), 28.0)

    # pesos por posición (ejemplo razonable)
    pos_w = {
        "GK":  WeightsR(0.25, 0.45, 0.30),
        "DEF": WeightsR(0.45, 0.35, 0.20),
        "MID": WeightsR(0.35, 0.45, 0.20),
        "FWD": WeightsR(0.40, 0.45, 0.15),
        "DEFAULT": WeightsR(0.40, 0.40, 0.20),
    }
    w = pos_w.get(position.upper(), pos_w["DEFAULT"])

    # alphas suben con s, betas bajan con s
    p = ODEParams(
        alphaF=0.25 + 0.25*s,
        alphaT=0.18 + 0.22*s,
        alphaM=0.12 + 0.18*s,
        betaF0=0.10 - 0.05*s,
        betaT0=0.08 - 0.04*s,
        betaM0=0.06 - 0.03*s,
        gammaFT=0.02 + 0.03*s,
        gammaFM=0.015 + 0.02*s,
        gammaTM=0.015 + 0.02*s,
        Aopt=Aopt,
        sigma=2.0,
        deltaM=0.20,
        slopeF=0.10,
        slopeT=0.08,
        slopeM=0.06
    )
    return p, w

# =========================
# Sistema EDO (F,T,M). R se calcula aparte
# =========================
def beta_age(age: float, beta0: float, pivot: float = 30.0, slope: float = 0.06) -> float:
    """
    Decaimiento aumenta después de pivot.
    Ej: a los 34, beta crece vs beta0.
    """
    if age <= pivot:
        return beta0
    return beta0 * (1.0 + slope * (age - pivot))

def ode_rhs(
    t: float,
    y: Tuple[float, float, float],
    age0: float,
    params: ODEParams,
    train: TrainingRegime,
    fatigue: float,
) -> Tuple[float, float, float]:
    """
    y = (F,T,M) en escala 0..100
    """
    F, T, M = y
    age = age0 + t  # t en años

    # decaimiento por edad (crece tras 30)
    bF = beta_age(age, params.betaF0, pivot=30.0, slope=params.slopeF)
    bT = beta_age(age, params.betaT0, pivot=30.0, slope=params.slopeT)
    bM = beta_age(age, params.betaM0, pivot=30.0, slope=params.slopeM)

    # fatiga reduce crecimiento y aumenta “pérdida efectiva”
    fatigue_factor = (1.0 - 0.35 * fatigue)  # 1.0 -> sin fatiga, baja con fatiga
    fatigue_penalty = 0.08 * fatigue          # aumenta caída

    # pico físico por edad óptima (gauss)
    peak = gaussian_peak(age, params.Aopt, params.sigma)

    # crecimiento tipo logístico (se acerca a 100)
    dF = params.alphaF * train.EF * fatigue_factor * (1 - F/params.Fmax) * (0.6 + 0.4*peak) \
         - (bF + fatigue_penalty) * (F/100.0) * 10.0 \
         + params.gammaFT * (T - F)/100.0 * 10.0 \
         + params.gammaFM * (M - F)/100.0 * 10.0

    dT = params.alphaT * train.ET * fatigue_factor * (1 - T/params.Tmax) \
            - (bT + 0.6 * fatigue_penalty) * (T/100.0) * 10.0 \
            + params.gammaFT * (F - T)/100.0 * 10.0 \
            + params.gammaTM * (M - T)/100.0 * 10.0

    # M responde a progreso + entrenamiento mental, con deltaM
    growth_signal = (abs(dF) + abs(dT)) * 0.05
    dM = params.alphaM * train.EM * (1 - M/params.Mmax) \
            - (bM + 0.4 * fatigue_penalty) * (M/100.0) * 10.0 \
            + params.deltaM * growth_signal

    return dF, dT, dM

def fatique_penalty(x: float) -> float:
    # backward-compatible alias kept for older references
    return x

# =========================
# RK4 manual
# =========================
def rk4_step(
    f: Callable[[float, Tuple[float,float,float]], Tuple[float,float,float]],
    t: float,
    y: Tuple[float,float,float],
    h: float
) -> Tuple[float,float,float]:
    k1 = f(t, y)
    y2 = (y[0] + 0.5*h*k1[0], y[1] + 0.5*h*k1[1], y[2] + 0.5*h*k1[2])
    k2 = f(t + 0.5*h, y2)
    y3 = (y[0] + 0.5*h*k2[0], y[1] + 0.5*h*k2[1], y[2] + 0.5*h*k2[2])
    k3 = f(t + 0.5*h, y3)
    y4 = (y[0] + h*k3[0], y[1] + h*k3[1], y[2] + h*k3[2])
    k4 = f(t + h, y4)

    yn = (
        y[0] + (h/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]),
        y[1] + (h/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]),
        y[2] + (h/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]),
    )
    return yn

# =========================
# Simulación completa (con lesiones y fatiga)
# =========================
def simulate_player(
    years: float,
    dt: float,
    age0: float,
    y0: Tuple[float,float,float],
    params: ODEParams,
    weights: WeightsR,
    train_regime: TrainingRegime,
    injuries: Optional[List[InjuryEvent]] = None,
    fatigue_cfg: Optional[FatigueConfig] = None,
    normalize_weights_for_R: bool = True
) -> Dict[str, List[float]]:
    """
    Retorna series: t, age, F, T, M, R, fatigue
    """
    injuries = injuries or []
    fatigue_cfg = fatigue_cfg or FatigueConfig(enabled=False)

    n = int(years / dt) + 1
    t = 0.0
    F, T, M = y0
    fatigue = 0.0

    out = {"t": [], "age": [], "F": [], "T": [], "M": [], "R": [], "fatigue": []}

    def injury_multiplier(time_years: float) -> float:
        # multiplica F si hay lesión
        mult = 1.0
        for ev in injuries:
            if ev.start_year <= time_years <= (ev.start_year + ev.duration_years):
                if ev.mode == "shock":
                    mult *= (1.0 - ev.severity)
                elif ev.mode == "exp_recovery":
                    # caída inmediata y recuperación exponencial dentro de la ventana
                    u = (time_years - ev.start_year) / max(ev.duration_years, 1e-9)
                    # al inicio ~ (1-sev), al final ~ 1.0
                    mult *= (1.0 - ev.severity * math.exp(-4.0*u))
                else:
                    mult *= (1.0 - ev.severity)
        return mult

    for _ in range(n):
        age = age0 + t

        # aplicar lesión solo a F (como pide el enunciado)
        F_eff = clamp(F * injury_multiplier(t), 0.0, 100.0)
        T_eff = clamp(T, 0.0, 100.0)
        M_eff = clamp(M, 0.0, 100.0)

        R = compute_R(F_eff, T_eff, M_eff, weights, normalize_weights=normalize_weights_for_R)

        out["t"].append(t)
        out["age"].append(age)
        out["F"].append(F_eff)
        out["T"].append(T_eff)
        out["M"].append(M_eff)
        out["R"].append(R)
        out["fatigue"].append(fatigue)

        # actualizar fatiga (simple y controlable)
        if fatigue_cfg.enabled:
            load = (train_regime.EF + train_regime.ET + train_regime.EM) / 3.0
            fatigue = clamp(fatigue + fatigue_cfg.k * load * dt - fatigue_cfg.recovery * (1.0 - load) * dt, 0.0, fatigue_cfg.cap)

        # rhs fijo para RK4
        def f(local_t: float, local_y: Tuple[float,float,float]):
            return ode_rhs(local_t, local_y, age0=age0, params=params, train=train_regime, fatigue=fatigue)

        F, T, M = rk4_step(f, t, (F, T, M), dt)

        # clamp estados
        F = clamp(F, 0.0, 100.0)
        T = clamp(T, 0.0, 100.0)
        M = clamp(M, 0.0, 100.0)

        t += dt

    return out
