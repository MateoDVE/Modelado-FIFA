# dashboard_streamlit.py
"""
Dashboard interactivo para simulaciones de futbolistas
Estudiante D: Sistema Integrado Final
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from edo_core import (
    TrainingRegime, WeightsR, ODEParams, InjuryEvent, FatigueConfig,
    simulate_player, calibrate_params_from_ml
)

# Configuración de página
st.set_page_config(
    page_title="Sistema de Análisis de Futbolistas",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚽ Sistema Dinámico de Análisis de Futbolistas")
st.markdown("### Modelado con Ecuaciones Diferenciales Ordinarias")

# =====================
# SIDEBAR - Configuración
# =====================
with st.sidebar:
    st.header("🎯 Configuración del Jugador")
    
    # Datos iniciales
    st.subheader("Datos Iniciales")
    age0 = st.slider("Edad inicial (años)", 16, 35, 22)
    F0 = st.slider("Físico inicial (F)", 40.0, 90.0, 65.0)
    T0 = st.slider("Técnica inicial (T)", 40.0, 90.0, 62.0)
    M0 = st.slider("Mental inicial (M)", 40.0, 90.0, 60.0)
    
    potential_pred = st.slider("Potencial predicho (ML)", 40.0, 99.0, 85.0)
    position = st.selectbox("Posición", ["DEFAULT", "GK", "DEF", "MID", "FWD"])
    
    st.subheader("⏱️ Simulación")
    years = st.slider("Años a simular", 1, 20, 10)
    dt = 1/52  # semanal
    
    st.subheader("🏋️ Régimen de Entrenamiento")
    train_mode = st.radio("Modo", ["Balanceado", "Físico", "Técnico", "Mental", "Custom"])
    
    if train_mode == "Balanceado":
        EF = ET = EM = st.slider("Intensidad balanceada", 0.0, 1.0, 0.7)
    elif train_mode == "Físico":
        EF = st.slider("Intensidad física", 0.0, 1.0, 0.9)
        ET = EM = 0.5
    elif train_mode == "Técnico":
        ET = st.slider("Intensidad técnica", 0.0, 1.0, 0.9)
        EF = EM = 0.5
    elif train_mode == "Mental":
        EM = st.slider("Intensidad mental", 0.0, 1.0, 0.9)
        EF = ET = 0.5
    else:
        EF = st.slider("EF - Físico", 0.0, 1.0, 0.7)
        ET = st.slider("ET - Técnico", 0.0, 1.0, 0.7)
        EM = st.slider("EM - Mental", 0.0, 1.0, 0.7)
    
    st.subheader("🤕 Lesiones")
    add_injury = st.checkbox("Agregar lesión")
    injuries = []
    if add_injury:
        injury_start = st.slider("Inicio lesión (años)", 0.0, float(years), 2.0)
        injury_duration = st.slider("Duración (meses)", 1, 12, 6) / 12.0
        injury_severity = st.slider("Severidad (%)", 10, 80, 50) / 100.0
        injury_mode = st.selectbox("Tipo recuperación", ["shock", "exp_recovery"])
        injuries = [InjuryEvent(injury_start, injury_duration, injury_severity, injury_mode)]
    
    st.subheader("😓 Fatiga")
    enable_fatigue = st.checkbox("Activar fatiga")
    if enable_fatigue:
        fatigue_k = st.slider("Tasa acumulación", 0.0, 0.5, 0.12)
        fatigue_recovery = st.slider("Tasa recuperación", 0.0, 0.5, 0.25)
        fatigue_cfg = FatigueConfig(True, fatigue_k, fatigue_recovery, 1.0)
    else:
        fatigue_cfg = FatigueConfig(False)

# =====================
# SIMULACIÓN
# =====================
# Calibrar parámetros
params, weights = calibrate_params_from_ml(potential_pred, position)

# Crear régimen
train_regime = TrainingRegime(EF, ET, EM)

# Simular
y0 = (F0, T0, M0)
sim = simulate_player(
    years=years,
    dt=dt,
    age0=age0,
    y0=y0,
    params=params,
    weights=weights,
    train_regime=train_regime,
    injuries=injuries,
    fatigue_cfg=fatigue_cfg
)

# =====================
# VISUALIZACIONES
# =====================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Evolución de Atributos (F, T, M)")
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=sim["age"], y=sim["F"], name="Físico (F)", 
                              line=dict(color='red', width=2)))
    fig1.add_trace(go.Scatter(x=sim["age"], y=sim["T"], name="Técnica (T)", 
                              line=dict(color='blue', width=2)))
    fig1.add_trace(go.Scatter(x=sim["age"], y=sim["M"], name="Mental (M)", 
                              line=dict(color='green', width=2)))
    
    fig1.update_layout(
        xaxis_title="Edad (años)",
        yaxis_title="Puntuación (0-100)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("⭐ Overall Rating (R)")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sim["age"], y=sim["R"], name="Rating (R)", 
                              fill='tozeroy', line=dict(color='purple', width=3)))
    
    # Marcar pico
    max_R_idx = np.argmax(sim["R"])
    max_R = sim["R"][max_R_idx]
    max_age = sim["age"][max_R_idx]
    
    fig2.add_trace(go.Scatter(
        x=[max_age], y=[max_R],
        mode='markers+text',
        marker=dict(size=12, color='gold', symbol='star'),
        text=[f"Pico: {max_R:.1f}"],
        textposition="top center",
        name="Pico"
    ))
    
    fig2.update_layout(
        xaxis_title="Edad (años)",
        yaxis_title="Overall Rating",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)

# Fatiga si está habilitada
if enable_fatigue:
    st.subheader("😓 Fatiga Acumulada")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sim["age"], y=sim["fatigue"], name="Fatiga",
                              fill='tozeroy', line=dict(color='orange', width=2)))
    fig3.update_layout(
        xaxis_title="Edad (años)",
        yaxis_title="Nivel de Fatiga (0-1)",
        height=250
    )
    st.plotly_chart(fig3, use_container_width=True)

# =====================
# MÉTRICAS CLAVE
# =====================
st.header("📈 Métricas Clave")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Rating Inicial", f"{sim['R'][0]:.1f}")
with col2:
    st.metric("Rating Final", f"{sim['R'][-1]:.1f}")
with col3:
    st.metric("Rating Máximo", f"{max_R:.1f}", f"@ {max_age:.1f} años")
with col4:
    change = sim['R'][-1] - sim['R'][0]
    st.metric("Cambio Total", f"{change:+.1f}", delta_color="normal")
with col5:
    avg_R = np.mean(sim["R"])
    st.metric("Rating Promedio", f"{avg_R:.1f}")

# =====================
# ANÁLISIS DETALLADO
# =====================
st.header("🔍 Análisis Detallado")

tab1, tab2, tab3 = st.tabs(["📊 Estadísticas", "📉 Tendencias", "🎯 Parámetros del Modelo"])

with tab1:
    st.subheader("Estadísticas por Atributo")
    
    stats_data = {
        "Atributo": ["Físico (F)", "Técnica (T)", "Mental (M)", "Rating (R)"],
        "Inicial": [sim["F"][0], sim["T"][0], sim["M"][0], sim["R"][0]],
        "Final": [sim["F"][-1], sim["T"][-1], sim["M"][-1], sim["R"][-1]],
        "Máximo": [max(sim["F"]), max(sim["T"]), max(sim["M"]), max(sim["R"])],
        "Mínimo": [min(sim["F"]), min(sim["T"]), min(sim["M"]), min(sim["R"])],
        "Promedio": [np.mean(sim["F"]), np.mean(sim["T"]), np.mean(sim["M"]), np.mean(sim["R"])],
        "Desv. Est.": [np.std(sim["F"]), np.std(sim["T"]), np.std(sim["M"]), np.std(sim["R"])]
    }
    
    df_stats = pd.DataFrame(stats_data)
    st.dataframe(df_stats.style.format("{:.2f}", subset=df_stats.columns[1:]), use_container_width=True)

with tab2:
    st.subheader("Tasas de Crecimiento/Decaimiento")
    
    # Calcular derivadas numéricas
    def compute_derivative(values, dt_years):
        return [(values[i+1] - values[i]) / dt_years for i in range(len(values)-1)]
    
    dF_dt = compute_derivative(sim["F"], dt)
    dT_dt = compute_derivative(sim["T"], dt)
    dM_dt = compute_derivative(sim["M"], dt)
    dR_dt = compute_derivative(sim["R"], dt)
    
    fig4 = make_subplots(rows=2, cols=2, 
                         subplot_titles=("dF/dt", "dT/dt", "dM/dt", "dR/dt"))
    
    fig4.add_trace(go.Scatter(x=sim["age"][:-1], y=dF_dt, name="dF/dt", 
                              line=dict(color='red')), row=1, col=1)
    fig4.add_trace(go.Scatter(x=sim["age"][:-1], y=dT_dt, name="dT/dt", 
                              line=dict(color='blue')), row=1, col=2)
    fig4.add_trace(go.Scatter(x=sim["age"][:-1], y=dM_dt, name="dM/dt", 
                              line=dict(color='green')), row=2, col=1)
    fig4.add_trace(go.Scatter(x=sim["age"][:-1], y=dR_dt, name="dR/dt", 
                              line=dict(color='purple')), row=2, col=2)
    
    fig4.update_xaxes(title_text="Edad", row=2, col=1)
    fig4.update_xaxes(title_text="Edad", row=2, col=2)
    fig4.update_yaxes(title_text="Tasa", row=1, col=1)
    fig4.update_yaxes(title_text="Tasa", row=1, col=2)
    
    fig4.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader("Parámetros del Modelo EDO")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Tasas de Aprendizaje (α)**")
        st.write(f"- αF (Físico): {params.alphaF:.3f}")
        st.write(f"- αT (Técnico): {params.alphaT:.3f}")
        st.write(f"- αM (Mental): {params.alphaM:.3f}")
        
        st.markdown("**Tasas de Decaimiento (β₀)**")
        st.write(f"- βF₀ (Físico): {params.betaF0:.3f}")
        st.write(f"- βT₀ (Técnico): {params.betaT0:.3f}")
        st.write(f"- βM₀ (Mental): {params.betaM0:.3f}")
        
        st.markdown("**Sinergias (γ)**")
        st.write(f"- γ(F↔T): {params.gammaFT:.3f}")
        st.write(f"- γ(F↔M): {params.gammaFM:.3f}")
        st.write(f"- γ(T↔M): {params.gammaTM:.3f}")
    
    with col2:
        st.markdown("**Pico Físico (Gaussiano)**")
        st.write(f"- Edad óptima (Aopt): {params.Aopt:.1f} años")
        st.write(f"- Anchura (σ): {params.sigma:.1f}")
        
        st.markdown("**Pesos del Rating (w)**")
        w_norm = weights.normalized()
        st.write(f"- wF (Físico): {w_norm.wF:.2f}")
        st.write(f"- wT (Técnico): {w_norm.wT:.2f}")
        st.write(f"- wM (Mental): {w_norm.wM:.2f}")
        
        st.markdown("**Otros Parámetros**")
        st.write(f"- δM (Sensibilidad Mental): {params.deltaM:.2f}")
        st.write(f"- Slope F (envejecimiento): {params.slopeF:.2f}")
        st.write(f"- Slope T (envejecimiento): {params.slopeT:.2f}")
        st.write(f"- Slope M (envejecimiento): {params.slopeM:.2f}")

# =====================
# EXPORTAR DATOS
# =====================
st.header("💾 Exportar Datos")

if st.button("📥 Descargar Simulación CSV"):
    df_export = pd.DataFrame({
        "Tiempo (años)": sim["t"],
        "Edad": sim["age"],
        "Físico (F)": sim["F"],
        "Técnica (T)": sim["T"],
        "Mental (M)": sim["M"],
        "Rating (R)": sim["R"],
        "Fatiga": sim["fatigue"]
    })
    
    csv = df_export.to_csv(index=False)
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name=f"simulacion_futbolista_edad{age0}_pot{potential_pred:.0f}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
**Sistema de Análisis de Futbolistas - Día 3: Integración Total y Sistema Dinámico**  
Modelado con Ecuaciones Diferenciales Ordinarias (EDO) y método RK4  
© 2025 - Práctica Integrada Avanzada
""")
