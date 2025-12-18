# sistema_integrado.py
"""
Sistema Integrado Completo: Día 2 (ML) + Día 3 (EDO)
Estudiante D: Integración Total

Pipeline: datos → redes neuronales → simulación dinámica → análisis
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime

# Importar componentes del Día 2
try:
    from models import create_mlp_regression, create_mlp_classification
    from preprocesamiento import cargar_y_limpiar_datos
except:
    print("Advertencia: No se pudieron importar modelos del Día 2")

# Importar componentes del Día 3
from edo_core import (
    TrainingRegime, WeightsR, ODEParams, InjuryEvent, FatigueConfig,
    simulate_player, calibrate_params_from_ml, compute_R
)


class SistemaIntegradoFutbolistas:
    """
    Sistema completo que integra:
    1. Predicción ML (Día 2): Overall y Potential
    2. Simulación EDO (Día 3): Evolución temporal
    3. Análisis y recomendaciones
    """
    
    def __init__(self, models_path: str = "Dia2/EntrenamientoResults"):
        self.models_path = Path(models_path)
        self.red1_model = None  # MLP Regresión (Overall)
        self.red2_model = None  # MLP Clasificación (Potential)
        self.scaler = None
        self.label_encoder = None
        
    def cargar_modelos_ml(self):
        """Carga los modelos entrenados del Día 2"""
        print("🔄 Cargando modelos ML del Día 2...")
        
        # Intentar cargar modelos si existen
        try:
            # Aquí deberías cargar tus modelos guardados
            # Por ahora, usamos simulación
            print("⚠️ Usando modelos simulados (implementar carga real)")
            self.red1_model = "RED1_PLACEHOLDER"
            self.red2_model = "RED2_PLACEHOLDER"
            return True
        except Exception as e:
            print(f"⚠️ Error cargando modelos: {e}")
            return False
    
    def predecir_desde_datos(self, player_data: Dict) -> Tuple[float, float]:
        """
        Predice Overall y Potential usando los modelos ML
        
        Args:
            player_data: Dict con features del jugador
            
        Returns:
            (overall_pred, potential_pred)
        """
        # Simulación - reemplazar con predicción real
        overall_pred = player_data.get('overall', 75.0)
        potential_pred = player_data.get('potential', 80.0)
        
        print(f"📊 Predicciones ML:")
        print(f"   Overall predicho: {overall_pred:.2f}")
        print(f"   Potential predicho: {potential_pred:.2f}")
        
        return overall_pred, potential_pred
    
    def extraer_estado_inicial(self, player_data: Dict, overall_pred: float) -> Tuple[float, float, float]:
        """
        Extrae o estima el estado inicial (F, T, M) del jugador
        
        Args:
            player_data: Datos del jugador
            overall_pred: Overall predicho por ML
            
        Returns:
            (F0, T0, M0) en escala 0-100
        """
        # Si tenemos datos específicos, usarlos
        if 'pace' in player_data and 'shooting' in player_data:
            # F: físico (pace, stamina, strength)
            F0 = np.mean([
                player_data.get('pace', 65),
                player_data.get('physic', 65)
            ])
            
            # T: técnica (shooting, passing, dribbling)
            T0 = np.mean([
                player_data.get('shooting', 62),
                player_data.get('passing', 62),
                player_data.get('dribbling', 62)
            ])
            
            # M: mental (vision, composure, positioning)
            M0 = overall_pred * 0.85  # aproximación
            
        else:
            # Estimación basada en overall
            base = overall_pred
            F0 = base - 5 + np.random.uniform(-3, 3)
            T0 = base - 3 + np.random.uniform(-3, 3)
            M0 = base - 8 + np.random.uniform(-3, 3)
        
        # Asegurar rango válido
        F0 = np.clip(F0, 40, 95)
        T0 = np.clip(T0, 40, 95)
        M0 = np.clip(M0, 40, 95)
        
        return F0, T0, M0
    
    def simular_carrera_completa(
        self,
        player_data: Dict,
        years: float = 15,
        training_scenario: str = "balanced",
        injuries_config: Optional[list] = None,
        enable_fatigue: bool = False
    ) -> Dict:
        """
        Pipeline completo: ML → Simulación EDO
        
        Args:
            player_data: Datos del jugador (dict con features)
            years: Años a simular
            training_scenario: "balanced", "intensive", "technical", "physical"
            injuries_config: Lista de eventos de lesión
            enable_fatigue: Activar modelo de fatiga
            
        Returns:
            Dict con resultados completos
        """
        print("\n" + "="*60)
        print("🚀 SISTEMA INTEGRADO - ANÁLISIS COMPLETO")
        print("="*60)
        
        # 1. Predicciones ML
        overall_pred, potential_pred = self.predecir_desde_datos(player_data)
        
        # 2. Estado inicial
        age0 = player_data.get('age', 22)
        position = player_data.get('position', 'DEFAULT')
        F0, T0, M0 = self.extraer_estado_inicial(player_data, overall_pred)
        
        print(f"\n📋 Perfil del Jugador:")
        print(f"   Edad: {age0} años")
        print(f"   Posición: {position}")
        print(f"   Estado inicial - F: {F0:.1f}, T: {T0:.1f}, M: {M0:.1f}")
        
        # 3. Calibrar parámetros EDO
        params, weights = calibrate_params_from_ml(potential_pred, position)
        
        # 4. Configurar entrenamiento
        train_regimes = {
            "balanced": TrainingRegime(0.7, 0.7, 0.7),
            "intensive": TrainingRegime(0.9, 0.9, 0.8),
            "technical": TrainingRegime(0.5, 0.9, 0.7),
            "physical": TrainingRegime(0.9, 0.6, 0.6),
            "conservative": TrainingRegime(0.5, 0.5, 0.5)
        }
        train_regime = train_regimes.get(training_scenario, train_regimes["balanced"])
        
        # 5. Configurar fatiga
        if enable_fatigue:
            fatigue_cfg = FatigueConfig(enabled=True, k=0.15, recovery=0.25, cap=1.0)
        else:
            fatigue_cfg = FatigueConfig(enabled=False)
        
        # 6. Simular
        print(f"\n⚙️ Simulando {years} años con régimen '{training_scenario}'...")
        
        dt = 1/52  # semanal
        y0 = (F0, T0, M0)
        
        sim = simulate_player(
            years=years,
            dt=dt,
            age0=age0,
            y0=y0,
            params=params,
            weights=weights,
            train_regime=train_regime,
            injuries=injuries_config or [],
            fatigue_cfg=fatigue_cfg
        )
        
        # 7. Análisis de resultados
        resultados = self._analizar_resultados(sim, player_data, overall_pred, potential_pred)
        
        # 8. Recomendaciones
        recomendaciones = self._generar_recomendaciones(sim, params, train_regime, player_data)
        
        return {
            "player_data": player_data,
            "predictions": {"overall": overall_pred, "potential": potential_pred},
            "initial_state": {"F0": F0, "T0": T0, "M0": M0, "age0": age0},
            "simulation": sim,
            "analysis": resultados,
            "recommendations": recomendaciones,
            "parameters": {
                "ode_params": params.__dict__,
                "weights": weights.__dict__,
                "training": train_regime.__dict__
            }
        }
    
    def _analizar_resultados(self, sim: Dict, player_data: Dict, overall_pred: float, potential_pred: float) -> Dict:
        """Análisis estadístico de la simulación"""
        
        R_vals = np.array(sim["R"])
        F_vals = np.array(sim["F"])
        T_vals = np.array(sim["T"])
        M_vals = np.array(sim["M"])
        ages = np.array(sim["age"])
        
        # Pico
        idx_peak = np.argmax(R_vals)
        R_peak = R_vals[idx_peak]
        age_peak = ages[idx_peak]
        
        # Desarrollo
        R_inicial = R_vals[0]
        R_final = R_vals[-1]
        desarrollo_total = R_final - R_inicial
        
        # ¿Alcanzó potencial?
        alcanzado_potencial = (R_peak >= potential_pred * 0.95)
        
        # Fase de declive
        decline_start_idx = idx_peak
        decline_rate = 0.0
        if decline_start_idx < len(R_vals) - 1:
            decline_years = ages[-1] - ages[decline_start_idx]
            if decline_years > 0:
                decline_rate = (R_vals[-1] - R_vals[decline_start_idx]) / decline_years
        
        # Estabilidad
        std_R = np.std(R_vals)
        
        analisis = {
            "peak_rating": R_peak,
            "peak_age": age_peak,
            "initial_rating": R_inicial,
            "final_rating": R_final,
            "total_development": desarrollo_total,
            "reached_potential": alcanzado_potencial,
            "decline_rate": decline_rate,
            "stability": std_R,
            "avg_rating": np.mean(R_vals),
            "best_attribute": max([("F", np.max(F_vals)), ("T", np.max(T_vals)), ("M", np.max(M_vals))], key=lambda x: x[1]),
            "career_years_analyzed": ages[-1] - ages[0]
        }
        
        print(f"\n📊 RESULTADOS DEL ANÁLISIS:")
        print(f"   Rating pico: {R_peak:.2f} @ {age_peak:.1f} años")
        print(f"   Desarrollo total: {desarrollo_total:+.2f}")
        print(f"   ¿Alcanzó potencial?: {'✓ Sí' if alcanzado_potencial else '✗ No'}")
        print(f"   Tasa de declive: {decline_rate:.2f}/año")
        
        return analisis
    
    def _generar_recomendaciones(self, sim: Dict, params: ODEParams, train: TrainingRegime, player_data: Dict) -> Dict:
        """Genera recomendaciones personalizadas"""
        
        recomendaciones = {
            "training": [],
            "career": [],
            "risk_management": []
        }
        
        # Análisis de entrenamiento
        if train.EF > 0.8:
            recomendaciones["training"].append(
                "⚠️ Entrenamiento físico muy intenso. Riesgo de lesiones. Considerar reducir a 0.7-0.75."
            )
        
        if train.ET < 0.6:
            recomendaciones["training"].append(
                "📈 La técnica puede mejorarse con mayor entrenamiento (aumentar ET a 0.7+)."
            )
        
        if train.EM < 0.5:
            recomendaciones["training"].append(
                "🧠 Aspecto mental desatendido. Incrementar EM para mejor rendimiento bajo presión."
            )
        
        # Análisis de carrera
        R_vals = np.array(sim["R"])
        ages = np.array(sim["age"])
        idx_peak = np.argmax(R_vals)
        age_peak = ages[idx_peak]
        
        if age_peak < 26:
            recomendaciones["career"].append(
                f"🌟 Pico temprano ({age_peak:.0f} años). Maximizar exposición en años prime."
            )
        elif age_peak > 30:
            recomendaciones["career"].append(
                f"⏱️ Desarrollo tardío. Pico a los {age_peak:.0f} años. Carrera más larga."
            )
        
        # Gestión de riesgo
        if params.betaF0 > 0.1:
            recomendaciones["risk_management"].append(
                "🏥 Alto decaimiento físico. Priorizar recuperación y prevención de lesiones."
            )
        
        if player_data.get('age', 22) > 28:
            recomendaciones["risk_management"].append(
                "📉 Jugador en fase madura. Ajustar intensidad y enfocarse en mantenimiento."
            )
        
        return recomendaciones
    
    def comparar_escenarios(self, player_data: Dict, scenarios: list) -> Dict:
        """
        Compara múltiples escenarios de entrenamiento
        
        Args:
            player_data: Datos del jugador
            scenarios: Lista de nombres de escenarios a comparar
            
        Returns:
            Dict con comparación
        """
        print("\n🔄 COMPARACIÓN DE ESCENARIOS")
        print("="*60)
        
        resultados_comparativos = {}
        
        for scenario in scenarios:
            print(f"\n📊 Simulando escenario: {scenario}")
            resultado = self.simular_carrera_completa(
                player_data=player_data,
                years=12,
                training_scenario=scenario,
                enable_fatigue=False
            )
            resultados_comparativos[scenario] = resultado
        
        # Comparativa
        print("\n📈 COMPARATIVA DE RESULTADOS:")
        print("-" * 60)
        for scenario, res in resultados_comparativos.items():
            analisis = res["analysis"]
            print(f"{scenario:15} | Pico: {analisis['peak_rating']:.1f} @ {analisis['peak_age']:.0f} años | "
                  f"Desarrollo: {analisis['total_development']:+.1f}")
        
        return resultados_comparativos
    
    def exportar_reporte(self, resultado: Dict, filename: str):
        """Exporta reporte completo en JSON"""
        output_path = Path("Dia3 EDO___sIMULACION") / "ResultadosIntegrados" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convertir arrays numpy a listas para JSON
        resultado_export = resultado.copy()
        sim = resultado_export["simulation"]
        for key in sim:
            if isinstance(sim[key], np.ndarray):
                sim[key] = sim[key].tolist()
            elif isinstance(sim[key], list) and len(sim[key]) > 0 and isinstance(sim[key][0], np.floating):
                sim[key] = [float(x) for x in sim[key]]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(resultado_export, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Reporte exportado: {output_path}")
    
    def visualizar_completo(self, resultado: Dict, save_path: Optional[str] = None):
        """Visualización completa de resultados"""
        sim = resultado["simulation"]
        analisis = resultado["analysis"]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Análisis Completo - {resultado['player_data'].get('name', 'Jugador')}", 
                     fontsize=16, fontweight='bold')
        
        # 1. F, T, M
        ax1 = axes[0, 0]
        ax1.plot(sim["age"], sim["F"], 'r-', label='Físico (F)', linewidth=2)
        ax1.plot(sim["age"], sim["T"], 'b-', label='Técnica (T)', linewidth=2)
        ax1.plot(sim["age"], sim["M"], 'g-', label='Mental (M)', linewidth=2)
        ax1.set_xlabel('Edad (años)')
        ax1.set_ylabel('Puntuación')
        ax1.set_title('Evolución de Atributos')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rating
        ax2 = axes[0, 1]
        ax2.plot(sim["age"], sim["R"], 'purple', linewidth=3)
        ax2.axhline(analisis["peak_rating"], color='gold', linestyle='--', 
                    label=f'Pico: {analisis["peak_rating"]:.1f}')
        ax2.scatter([analisis["peak_age"]], [analisis["peak_rating"]], 
                    color='gold', s=200, marker='*', zorder=5)
        ax2.set_xlabel('Edad (años)')
        ax2.set_ylabel('Overall Rating')
        ax2.set_title('Overall Rating (R)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Comparativa inicial vs pico vs final
        ax3 = axes[1, 0]
        attrs = ['F', 'T', 'M', 'R']
        inicial = [sim["F"][0], sim["T"][0], sim["M"][0], sim["R"][0]]
        pico_idx = np.argmax(sim["R"])
        pico = [sim["F"][pico_idx], sim["T"][pico_idx], sim["M"][pico_idx], sim["R"][pico_idx]]
        final = [sim["F"][-1], sim["T"][-1], sim["M"][-1], sim["R"][-1]]
        
        x = np.arange(len(attrs))
        width = 0.25
        ax3.bar(x - width, inicial, width, label='Inicial', alpha=0.8)
        ax3.bar(x, pico, width, label='Pico', alpha=0.8)
        ax3.bar(x + width, final, width, label='Final', alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(attrs)
        ax3.set_ylabel('Valor')
        ax3.set_title('Comparativa: Inicial vs Pico vs Final')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Métricas clave (texto)
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        info_text = f"""
        MÉTRICAS CLAVE:
        
        • Rating Pico: {analisis['peak_rating']:.2f}
        • Edad Pico: {analisis['peak_age']:.1f} años
        
        • Rating Inicial: {analisis['initial_rating']:.2f}
        • Rating Final: {analisis['final_rating']:.2f}
        
        • Desarrollo Total: {analisis['total_development']:+.2f}
        • Tasa Declive: {analisis['decline_rate']:.3f}/año
        
        • ¿Alcanzó Potencial?: {'Sí ✓' if analisis['reached_potential'] else 'No ✗'}
        
        • Mejor Atributo: {analisis['best_attribute'][0]} ({analisis['best_attribute'][1]:.1f})
        
        • Años Analizados: {analisis['career_years_analyzed']:.1f}
        """
        
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualización guardada: {save_path}")
        
        plt.show()


# ======================
# FUNCIONES DE UTILIDAD
# ======================

def ejemplo_uso_basico():
    """Ejemplo básico de uso del sistema integrado"""
    
    # Inicializar sistema
    sistema = SistemaIntegradoFutbolistas()
    
    # Datos de ejemplo de un jugador
    jugador = {
        "name": "Joven Promesa",
        "age": 20,
        "position": "MID",
        "overall": 72,
        "potential": 88,
        "pace": 78,
        "shooting": 68,
        "passing": 74,
        "dribbling": 76,
        "physic": 70
    }
    
    # Simular carrera
    resultado = sistema.simular_carrera_completa(
        player_data=jugador,
        years=15,
        training_scenario="balanced",
        enable_fatigue=False
    )
    
    # Visualizar
    sistema.visualizar_completo(resultado)
    
    # Exportar
    sistema.exportar_reporte(resultado, "ejemplo_joven_promesa.json")
    
    return resultado


def ejemplo_comparacion_escenarios():
    """Ejemplo de comparación de escenarios"""
    
    sistema = SistemaIntegradoFutbolistas()
    
    jugador = {
        "name": "Jugador Versátil",
        "age": 23,
        "position": "FWD",
        "overall": 78,
        "potential": 85
    }
    
    escenarios = ["balanced", "intensive", "technical", "conservative"]
    
    comparacion = sistema.comparar_escenarios(jugador, escenarios)
    
    return comparacion


if __name__ == "__main__":
    print("🚀 Sistema Integrado de Análisis de Futbolistas")
    print("="*60)
    
    # Ejecutar ejemplo básico
    resultado = ejemplo_uso_basico()
    
    print("\n✅ Sistema integrado ejecutado exitosamente")
