"""Streamlit dashboard to interactively run the EDO simulator (RK4).

Run: `streamlit run EDO___sIMULACION/dashboard_streamlit.py`

Features:
- Adjust ODE params (via potential or manual sliders)
- Choose training regime and injury events
- Run RK4 simulation and visualize F,T,M,R over time
- Download simulation as CSV
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from EDO___sIMULACION import edo_core


st.set_page_config(page_title="Simulador EDO - RK4", layout="wide")

st.title("Simulador de Desarrollo de Futbolistas (RK4)")

with st.sidebar.form(key='params'):
    st.header("Parámetros generales")
    age0 = st.number_input("Edad inicial (años)", value=18.0, min_value=14.0, max_value=40.0, step=0.5)
    years = st.number_input("Duración (años)", value=10.0, min_value=1.0, max_value=20.0, step=0.5)
    dt = st.number_input("Paso dt (años)", value=1/12, format="%f")

    st.markdown("---")
    st.subheader("Calibración desde ML")
    potential = st.slider("Potential (ML) mapa 40..99", min_value=40, max_value=99, value=75)
    position = st.selectbox("Posición (calib) ", options=["DEFAULT","GK","DEF","MID","FWD"], index=0)

    st.markdown("---")
    st.subheader("Regímenes de entrenamiento")
    EF = st.slider("Entrenamiento Físico EF", 0.0, 1.0, 0.7)
    ET = st.slider("Entrenamiento Técnico ET", 0.0, 1.0, 0.7)
    EM = st.slider("Entrenamiento Mental EM", 0.0, 1.0, 0.7)

    st.markdown("---")
    st.subheader("Lesión (opcional)")
    inj_on = st.checkbox("Agregar lesión de ejemplo", value=False)
    inj_start = st.number_input("Lesión - inicio (años desde inicio)", value=2.0, min_value=0.0, max_value=years)
    inj_dur = st.number_input("Lesión - duración (años)", value=0.5, min_value=0.0, max_value=5.0)
    inj_sev = st.slider("Lesión - severidad (fracción)", 0.0, 1.0, 0.5)

    st.markdown("---")
    st.subheader("Condiciones iniciales (opcional)")
    F0_in = st.number_input("F0 (Físico)", value=50.0, min_value=0.0, max_value=100.0)
    T0_in = st.number_input("T0 (Técnico)", value=50.0, min_value=0.0, max_value=100.0)
    M0_in = st.number_input("M0 (Mental)", value=50.0, min_value=0.0, max_value=100.0)

    st.form_submit_button("Actualizar parámetros")


# --- Prepare params
params, weights = edo_core.calibrate_params_from_ml(potential, position)
train = edo_core.TrainingRegime(EF=EF, ET=ET, EM=EM)

injuries = []
if inj_on and inj_dur > 0:
    injuries.append(edo_core.InjuryEvent(start_year=inj_start, duration_years=inj_dur, severity=inj_sev, mode='exp_recovery'))

fatigue_cfg = edo_core.FatigueConfig(enabled=True, k=0.12, recovery=0.25, cap=1.0)

# initial state: derive from ML potential mapping (simple heuristic)
F0 = F0_in
T0 = T0_in
M0 = M0_in

cols1, cols2 = st.columns([2, 3])
with cols1:
    st.subheader("Pesos R (wF, wT, wM)")
    wF = st.slider("wF", 0.0, 1.0, float(weights.wF))
    wT = st.slider("wT", 0.0, 1.0, float(weights.wT))
    wM = st.slider("wM", 0.0, 1.0, float(weights.wM))
    weights = edo_core.WeightsR(wF, wT, wM)

with cols2:
    st.markdown("\n")
    st.markdown("### Escenarios rápidos")
    scenario = st.selectbox("Escenario", options=["Joven promesa","Lento pero seguro","Early bloomer","Late bloomer","Personalizado"])
    if scenario == "Joven promesa":
        F0, T0, M0 = 50.0, 45.0, 40.0
        EF, ET, EM = 0.9, 0.8, 0.6
    elif scenario == "Lento pero seguro":
        F0, T0, M0 = 40.0, 50.0, 50.0
        EF, ET, EM = 0.6, 0.6, 0.6
    elif scenario == "Early bloomer":
        F0, T0, M0 = 60.0, 55.0, 45.0
        EF, ET, EM = 0.8, 0.6, 0.5
    elif scenario == "Late bloomer":
        F0, T0, M0 = 35.0, 45.0, 50.0
        EF, ET, EM = 0.5, 0.7, 0.6

run = st.button("Run Simulation")
if run:
    sim = edo_core.simulate_player(
        years=float(years),
        dt=float(dt),
        age0=float(age0),
        y0=(float(F0), float(T0), float(M0)),
        params=params,
        weights=weights,
        train_regime=train,
        injuries=injuries,
        fatigue_cfg=fatigue_cfg,
        normalize_weights_for_R=True
    )

    df = pd.DataFrame({
        't': sim['t'],
        'age': sim['age'],
        'F': sim['F'],
        'T': sim['T'],
        'M': sim['M'],
        'R': sim['R'],
        'fatigue': sim['fatigue']
    })

    st.subheader("Resultados (serie temporal)")
    st.dataframe(df.head(200))

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['age'], y=df['F'], mode='lines', name='F (Físico)'))
    fig.add_trace(go.Scatter(x=df['age'], y=df['T'], mode='lines', name='T (Técnico)'))
    fig.add_trace(go.Scatter(x=df['age'], y=df['M'], mode='lines', name='M (Mental)'))
    fig.add_trace(go.Scatter(x=df['age'], y=df['R'], mode='lines', name='R (Overall)', line=dict(width=3, dash='dash')))

    fig.update_layout(title='Evolución por edad', xaxis_title='Edad', yaxis_title='Valor (0-100)')
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(label='Descargar CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='simulation.csv', mime='text/csv')

    st.markdown("---")
    st.subheader("Estadísticas finales")
    st.write(df.tail(1).T)

    st.info("Simulación completada. Ajusta parámetros y vuelve a ejecutar.")


# -------------------------
# Análisis de sensibilidad
# -------------------------
with st.expander("Análisis de sensibilidad (barrido de parámetros)", expanded=False):
    st.write("Explora cómo cambia la trayectoria de `R` al variar un parámetro de entrenamiento.")
    sweep_param = st.selectbox("Parámetro a barrer", options=['EF','ET','EM','All'], index=0)
    sweep_min = st.number_input("Valor mínimo", value=0.0, min_value=0.0, max_value=1.0, step=0.05)
    sweep_max = st.number_input("Valor máximo", value=1.0, min_value=0.0, max_value=1.0, step=0.05)
    sweep_steps = st.slider("Pasos (líneas)", min_value=3, max_value=21, value=7)
    run_sweep = st.button("Ejecutar análisis de sensibilidad")

    if run_sweep:
        values = list(np.linspace(sweep_min, sweep_max, sweep_steps))
        all_dfs = []
        fig_s = go.Figure()

        for v in values:
            # copy train regime
            tr = edo_core.TrainingRegime(EF=EF, ET=ET, EM=EM)
            if sweep_param == 'EF':
                tr.EF = float(v)
            elif sweep_param == 'ET':
                tr.ET = float(v)
            elif sweep_param == 'EM':
                tr.EM = float(v)
            else:  # All: scale all together
                tr.EF = tr.ET = tr.EM = float(v)

            sim_s = edo_core.simulate_player(
                years=float(years), dt=float(dt), age0=float(age0),
                y0=(float(F0), float(T0), float(M0)), params=params,
                weights=weights, train_regime=tr, injuries=injuries,
                fatigue_cfg=fatigue_cfg, normalize_weights_for_R=True)

            df_s = pd.DataFrame(sim_s)
            df_s['sweep_value'] = float(v)
            all_dfs.append(df_s)

            fig_s.add_trace(go.Scatter(x=df_s['age'], y=df_s['R'], mode='lines', name=f'{sweep_param}={v:.2f}'))

        st.plotly_chart(fig_s, use_container_width=True)

        big_df = pd.concat(all_dfs, ignore_index=True)
        st.download_button(label='Descargar resultados del barrido (CSV)', data=big_df.to_csv(index=False).encode('utf-8'), file_name='sensitivity_sweep.csv', mime='text/csv')

        st.success('Análisis de sensibilidad completado.')
