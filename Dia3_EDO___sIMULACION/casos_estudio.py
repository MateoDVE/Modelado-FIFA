# casos_estudio.py
"""
5 Casos de Estudio Completos
Estudiante D: Análisis de jugadores reales con sistema integrado
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sistema_integrado import SistemaIntegradoFutbolistas
from edo_core import InjuryEvent, TrainingRegime


# ======================
# DEFINICIÓN DE CASOS
# ======================

CASOS_ESTUDIO = {
    "caso_1_joven_promesa": {
        "name": "Joven Promesa Elite",
        "age": 19,
        "position": "FWD",
        "overall": 75,
        "potential": 92,
        "pace": 88,
        "shooting": 72,
        "passing": 68,
        "dribbling": 82,
        "physic": 70,
        "description": "Delantero joven con altísimo potencial. Físicamente rápido pero necesita desarrollo técnico.",
        "training_scenario": "intensive",
        "injuries": []
    },
    
    "caso_2_late_bloomer": {
        "name": "Desarrollo Tardío",
        "age": 24,
        "position": "MID",
        "overall": 72,
        "potential": 82,
        "pace": 70,
        "shooting": 68,
        "passing": 74,
        "dribbling": 72,
        "physic": 68,
        "description": "Centrocampista de desarrollo lento pero consistente. Pico esperado cerca de los 28-30.",
        "training_scenario": "balanced",
        "injuries": []
    },
    
    "caso_3_portero_veterano": {
        "name": "Guardameta Experimentado",
        "age": 28,
        "position": "GK",
        "overall": 84,
        "potential": 87,
        "pace": 45,
        "shooting": 20,
        "passing": 68,
        "dribbling": 30,
        "physic": 78,
        "description": "Portero en su prime. Los porteros tienen carreras más largas, pico tardío.",
        "training_scenario": "conservative",
        "injuries": []
    },
    
    "caso_4_defensa_lesiones": {
        "name": "Defensa con Historial de Lesiones",
        "age": 26,
        "position": "DEF",
        "overall": 80,
        "potential": 83,
        "pace": 72,
        "shooting": 45,
        "passing": 70,
        "dribbling": 65,
        "physic": 82,
        "description": "Defensa sólido pero propenso a lesiones. Requiere manejo cuidadoso.",
        "training_scenario": "conservative",
        "injuries": [
            InjuryEvent(start_year=1.5, duration_years=0.5, severity=0.4, mode="exp_recovery"),
            InjuryEvent(start_year=4.0, duration_years=0.33, severity=0.3, mode="shock"),
            InjuryEvent(start_year=7.5, duration_years=0.66, severity=0.5, mode="exp_recovery")
        ]
    },
    
    "caso_5_estrella_declive": {
        "name": "Estrella en Declive",
        "age": 32,
        "position": "FWD",
        "overall": 88,
        "potential": 88,
        "pace": 75,
        "shooting": 89,
        "passing": 82,
        "dribbling": 84,
        "physic": 76,
        "description": "Delantero estrella en fase de declive. Ya alcanzó su pico, objetivo: mantener nivel.",
        "training_scenario": "conservative",
        "injuries": []
    }
}


# ======================
# ANÁLISIS DE CASOS
# ======================

class AnalizadorCasosEstudio:
    """Ejecuta y documenta los 5 casos de estudio"""
    
    def __init__(self):
        self.sistema = SistemaIntegradoFutbolistas()
        self.resultados = {}
        self.output_dir = Path("Dia3 EDO___sIMULACION") / "CasosEstudio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def ejecutar_caso(self, caso_id: str, caso_data: dict, years: int = 12) -> dict:
        """Ejecuta un caso de estudio completo"""
        
        print("\n" + "="*80)
        print(f"📊 CASO DE ESTUDIO: {caso_id.upper()}")
        print("="*80)
        print(f"Nombre: {caso_data['name']}")
        print(f"Descripción: {caso_data['description']}")
        print(f"Edad: {caso_data['age']} años | Posición: {caso_data['position']}")
        print(f"Overall: {caso_data['overall']} | Potential: {caso_data['potential']}")
        
        # Simular
        resultado = self.sistema.simular_carrera_completa(
            player_data=caso_data,
            years=years,
            training_scenario=caso_data['training_scenario'],
            injuries_config=caso_data.get('injuries', []),
            enable_fatigue=True
        )
        
        # Imprimir recomendaciones
        print("\n💡 RECOMENDACIONES:")
        recs = resultado['recommendations']
        
        if recs['training']:
            print("\n  Entrenamiento:")
            for r in recs['training']:
                print(f"    • {r}")
        
        if recs['career']:
            print("\n  Carrera:")
            for r in recs['career']:
                print(f"    • {r}")
        
        if recs['risk_management']:
            print("\n  Gestión de Riesgo:")
            for r in recs['risk_management']:
                print(f"    • {r}")
        
        return resultado
    
    def ejecutar_todos(self):
        """Ejecuta todos los casos de estudio"""
        
        print("\n" + "🎯"*40)
        print("EJECUTANDO 5 CASOS DE ESTUDIO COMPLETOS")
        print("🎯"*40)
        
        for caso_id, caso_data in CASOS_ESTUDIO.items():
            resultado = self.ejecutar_caso(caso_id, caso_data)
            self.resultados[caso_id] = resultado
            
            # Guardar individual
            filename = f"{caso_id}_resultado.json"
            self.sistema.exportar_reporte(resultado, filename)
            
            # Visualizar y guardar
            viz_path = self.output_dir / f"{caso_id}_visualizacion.png"
            self.sistema.visualizar_completo(resultado, save_path=str(viz_path))
        
        print("\n" + "="*80)
        print("✅ TODOS LOS CASOS COMPLETADOS")
        print("="*80)
    
    def generar_comparativa(self):
        """Genera un análisis comparativo de los 5 casos"""
        
        print("\n📊 ANÁLISIS COMPARATIVO DE CASOS")
        print("="*80)
        
        comparativa = []
        
        for caso_id, resultado in self.resultados.items():
            caso_data = CASOS_ESTUDIO[caso_id]
            analisis = resultado['analysis']
            
            comparativa.append({
                "caso": caso_id,
                "nombre": caso_data['name'],
                "edad_inicial": caso_data['age'],
                "posicion": caso_data['position'],
                "overall_inicial": caso_data['overall'],
                "potential": caso_data['potential'],
                "rating_pico": analisis['peak_rating'],
                "edad_pico": analisis['peak_age'],
                "desarrollo_total": analisis['total_development'],
                "alcanzó_potencial": analisis['reached_potential'],
                "tasa_declive": analisis['decline_rate']
            })
        
        # Imprimir tabla
        print(f"\n{'Caso':<30} | {'Edad':<4} | {'Pos':<4} | {'Pico R':<7} | {'@ Edad':<7} | {'Desarrollo':<10} | {'¿Potencial?'}")
        print("-" * 100)
        
        for c in comparativa:
            potencial_str = "✓ Sí" if c['alcanzó_potencial'] else "✗ No"
            print(f"{c['nombre']:<30} | {c['edad_inicial']:<4} | {c['posicion']:<4} | "
                  f"{c['rating_pico']:>7.2f} | {c['edad_pico']:>7.1f} | "
                  f"{c['desarrollo_total']:>+10.2f} | {potencial_str}")
        
        # Guardar comparativa
        comparativa_path = self.output_dir / "comparativa_casos.json"
        with open(comparativa_path, 'w', encoding='utf-8') as f:
            json.dump(comparativa, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Comparativa guardada: {comparativa_path}")
        
        # Visualización comparativa
        self._visualizar_comparativa(comparativa)
        
        return comparativa
    
    def _visualizar_comparativa(self, comparativa: list):
        """Crea visualización comparativa de los 5 casos"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Comparativa de 5 Casos de Estudio", fontsize=16, fontweight='bold')
        
        nombres = [c['nombre'][:20] for c in comparativa]
        
        # 1. Rating Pico
        ax1 = axes[0, 0]
        picos = [c['rating_pico'] for c in comparativa]
        colors = ['green' if c['alcanzó_potencial'] else 'orange' for c in comparativa]
        ax1.barh(nombres, picos, color=colors, alpha=0.7)
        ax1.set_xlabel('Rating Pico')
        ax1.set_title('Rating Máximo Alcanzado')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 2. Edad del Pico
        ax2 = axes[0, 1]
        edades_pico = [c['edad_pico'] for c in comparativa]
        ax2.barh(nombres, edades_pico, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Edad (años)')
        ax2.set_title('Edad del Pico de Rendimiento')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Desarrollo Total
        ax3 = axes[0, 2]
        desarrollos = [c['desarrollo_total'] for c in comparativa]
        colors_dev = ['green' if d > 0 else 'red' for d in desarrollos]
        ax3.barh(nombres, desarrollos, color=colors_dev, alpha=0.7)
        ax3.set_xlabel('Cambio en Rating')
        ax3.set_title('Desarrollo Total (Final - Inicial)')
        ax3.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Evolución temporal de todos
        ax4 = axes[1, 0]
        for caso_id, resultado in self.resultados.items():
            sim = resultado['simulation']
            nombre = CASOS_ESTUDIO[caso_id]['name'][:15]
            ax4.plot(sim['age'], sim['R'], label=nombre, linewidth=2)
        ax4.set_xlabel('Edad (años)')
        ax4.set_ylabel('Rating')
        ax4.set_title('Trayectorias de Rating')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Tasa de Declive
        ax5 = axes[1, 1]
        tasas_declive = [c['tasa_declive'] for c in comparativa]
        colors_dec = ['red' if t < -1 else 'orange' if t < 0 else 'green' for t in tasas_declive]
        ax5.barh(nombres, tasas_declive, color=colors_dec, alpha=0.7)
        ax5.set_xlabel('Rating/Año')
        ax5.set_title('Tasa de Declive (Post-Pico)')
        ax5.axvline(0, color='black', linestyle='-', linewidth=0.8)
        ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. Resumen por posición
        ax6 = axes[1, 2]
        posiciones = {}
        for c in comparativa:
            pos = c['posicion']
            if pos not in posiciones:
                posiciones[pos] = []
            posiciones[pos].append(c['rating_pico'])
        
        pos_names = list(posiciones.keys())
        pos_avg = [np.mean(posiciones[p]) for p in pos_names]
        ax6.bar(pos_names, pos_avg, color='purple', alpha=0.7)
        ax6.set_ylabel('Rating Pico Promedio')
        ax6.set_title('Rating Pico por Posición')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        viz_path = self.output_dir / "comparativa_visualizacion.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualización comparativa guardada: {viz_path}")
        
        plt.show()
    
    def generar_reporte_casos(self):
        """Genera un reporte en Markdown con todos los casos"""
        
        reporte_path = self.output_dir / "REPORTE_CASOS_ESTUDIO.md"
        
        with open(reporte_path, 'w', encoding='utf-8') as f:
            f.write("# 📊 Reporte de Casos de Estudio\n\n")
            f.write("## Sistema Integrado de Análisis de Futbolistas\n\n")
            f.write("---\n\n")
            
            f.write("## Resumen Ejecutivo\n\n")
            f.write(f"Se analizaron **5 casos de estudio** representativos utilizando el sistema integrado ")
            f.write("que combina predicciones de Machine Learning (Día 2) con simulaciones dinámicas ")
            f.write("mediante ecuaciones diferenciales ordinarias (Día 3).\n\n")
            
            f.write("### Casos Analizados:\n\n")
            for i, (caso_id, caso_data) in enumerate(CASOS_ESTUDIO.items(), 1):
                f.write(f"{i}. **{caso_data['name']}** - {caso_data['description']}\n")
            
            f.write("\n---\n\n")
            
            # Detalle de cada caso
            for caso_id, caso_data in CASOS_ESTUDIO.items():
                resultado = self.resultados.get(caso_id)
                if not resultado:
                    continue
                
                analisis = resultado['analysis']
                recs = resultado['recommendations']
                
                f.write(f"## 📋 Caso: {caso_data['name']}\n\n")
                
                f.write(f"**Descripción:** {caso_data['description']}\n\n")
                
                f.write("### Perfil Inicial\n\n")
                f.write(f"- **Edad:** {caso_data['age']} años\n")
                f.write(f"- **Posición:** {caso_data['position']}\n")
                f.write(f"- **Overall:** {caso_data['overall']}\n")
                f.write(f"- **Potential:** {caso_data['potential']}\n")
                f.write(f"- **Régimen de Entrenamiento:** {caso_data['training_scenario']}\n\n")
                
                f.write("### Resultados de la Simulación\n\n")
                f.write(f"- **Rating Pico:** {analisis['peak_rating']:.2f} (@ {analisis['peak_age']:.1f} años)\n")
                f.write(f"- **Rating Inicial:** {analisis['initial_rating']:.2f}\n")
                f.write(f"- **Rating Final:** {analisis['final_rating']:.2f}\n")
                f.write(f"- **Desarrollo Total:** {analisis['total_development']:+.2f}\n")
                f.write(f"- **¿Alcanzó Potencial?:** {'✓ Sí' if analisis['reached_potential'] else '✗ No'}\n")
                f.write(f"- **Tasa de Declive:** {analisis['decline_rate']:.3f} puntos/año\n")
                f.write(f"- **Mejor Atributo:** {analisis['best_attribute'][0]} ({analisis['best_attribute'][1]:.1f})\n\n")
                
                if len(caso_data.get('injuries', [])) > 0:
                    f.write(f"- **Lesiones Simuladas:** {len(caso_data['injuries'])}\n\n")
                
                f.write("### Recomendaciones\n\n")
                
                if recs['training']:
                    f.write("**Entrenamiento:**\n\n")
                    for r in recs['training']:
                        f.write(f"- {r}\n")
                    f.write("\n")
                
                if recs['career']:
                    f.write("**Carrera:**\n\n")
                    for r in recs['career']:
                        f.write(f"- {r}\n")
                    f.write("\n")
                
                if recs['risk_management']:
                    f.write("**Gestión de Riesgo:**\n\n")
                    for r in recs['risk_management']:
                        f.write(f"- {r}\n")
                    f.write("\n")
                
                f.write("### Gráficas\n\n")
                f.write(f"![Visualización de {caso_data['name']}]({caso_id}_visualizacion.png)\n\n")
                
                f.write("---\n\n")
            
            # Conclusiones
            f.write("## 🎯 Conclusiones Generales\n\n")
            
            f.write("### Hallazgos Principales:\n\n")
            f.write("1. **Desarrollo Diferenciado por Edad:** Los jugadores jóvenes muestran mayor potencial ")
            f.write("de crecimiento, mientras que los veteranos requieren estrategias de mantenimiento.\n\n")
            
            f.write("2. **Impacto de las Lesiones:** Las lesiones tienen efecto significativo en la trayectoria, ")
            f.write("especialmente si ocurren cerca del pico de rendimiento.\n\n")
            
            f.write("3. **Variación por Posición:** Los porteros (GK) muestran picos más tardíos y carreras ")
            f.write("más longevas comparado con delanteros (FWD) y mediocampistas (MID).\n\n")
            
            f.write("4. **Importancia del Régimen de Entrenamiento:** La intensidad y balance del entrenamiento ")
            f.write("determina la velocidad de desarrollo y el riesgo de fatiga/lesiones.\n\n")
            
            f.write("5. **Predicción vs Realidad:** El sistema integrado permite validar si las predicciones ML ")
            f.write("se materializan bajo diferentes condiciones de entrenamiento y eventos imprevistos.\n\n")
            
            f.write("---\n\n")
            f.write("*Reporte generado automáticamente por el Sistema Integrado de Análisis de Futbolistas*\n")
        
        print(f"\n📄 Reporte de casos guardado: {reporte_path}")
        return reporte_path


# ======================
# MAIN
# ======================

def main():
    """Ejecuta el análisis completo de los 5 casos de estudio"""
    
    print("\n" + "⚽"*40)
    print("ANÁLISIS DE 5 CASOS DE ESTUDIO")
    print("Sistema Integrado - Día 3")
    print("⚽"*40 + "\n")
    
    # Crear analizador
    analizador = AnalizadorCasosEstudio()
    
    # Ejecutar todos los casos
    analizador.ejecutar_todos()
    
    # Generar comparativa
    comparativa = analizador.generar_comparativa()
    
    # Generar reporte
    reporte_path = analizador.generar_reporte_casos()
    
    print("\n" + "="*80)
    print("✅ ANÁLISIS DE CASOS DE ESTUDIO COMPLETADO")
    print("="*80)
    print(f"\n📂 Resultados guardados en: {analizador.output_dir}")
    print(f"📄 Reporte principal: {reporte_path}")
    print("\n🎉 ¡Todos los entregables del Día 3 están listos!")


if __name__ == "__main__":
    main()
