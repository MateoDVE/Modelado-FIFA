"""
Análisis de Interpretabilidad con SHAP
======================================

Este script realiza análisis de importancia de características usando SHAP
(SHapley Additive exPlanations) para ambas redes neuronales.

SHAP proporciona:
- Importancia global de características
- Explicaciones locales para predicciones individuales
- Gráficos de dependencia
- Valores de contribución para cada característica

Autor: Estudiante B
Fecha: Diciembre 2025
"""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import json
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from models import (
    build_position_target_7_classes, undersample_balance, ID_COLS
)


# =========================
# INSTALACIÓN DE SHAP
# =========================
print("=" * 80)
print("📦 VERIFICANDO INSTALACIÓN DE SHAP")
print("=" * 80)

try:
    import shap
    print("✅ SHAP ya está instalado")
except ImportError:
    print("⚠️  SHAP no está instalado. Instalando...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    import shap
    print("✅ SHAP instalado correctamente")


# =========================
# CARGAR MODELOS Y DATOS
# =========================

def load_models_and_data():
    """
    Carga los modelos entrenados y el dataset
    """
    print("\n" + "=" * 80)
    print("📂 CARGANDO MODELOS Y DATOS")
    print("=" * 80)
    
    # Cargar modelos
    try:
        with open("red1_regresion_trained.pkl", "rb") as f:
            red1_data = pickle.load(f)
        print("✅ Red 1 (Regresión) cargada")
    except FileNotFoundError:
        print("❌ Error: red1_regresion_trained.pkl no encontrado")
        print("   Ejecuta primero: python estudiante_a_entrenamiento.py")
        return None, None, None
    
    try:
        with open("red2_clasificacion_trained.pkl", "rb") as f:
            red2_data = pickle.load(f)
        print("✅ Red 2 (Clasificación) cargada")
    except FileNotFoundError:
        print("❌ Error: red2_clasificacion_trained.pkl no encontrado")
        print("   Ejecuta primero: python estudiante_a_entrenamiento.py")
        return None, None, None
    
    # Cargar dataset
    script_dir = Path(__file__).parent
    DB_OPTIONS = [
        script_dir / "database_clean.sqlite",
        script_dir.parent / "database_clean.sqlite",
        Path("database_clean.sqlite")
    ]
    
    DB = None
    for db_path in DB_OPTIONS:
        if db_path.exists():
            DB = str(db_path)
            break
    
    if DB is None:
        print("❌ Error: database_clean.sqlite no encontrado")
        return None, None, None
    
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM player_attributes_clean;", conn)
    conn.close()
    
    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    return red1_data, red2_data, df


# =========================
# ANÁLISIS SHAP - RED 1 (REGRESIÓN)
# =========================

def analyze_shap_red1(model_data, df, n_samples=1000):
    """
    Análisis SHAP para Red 1 (Predicción de Potencial)
    
    Parámetros:
    -----------
    model_data : dict
        Diccionario con 'model', 'features', 'best_params'
    df : DataFrame
        Dataset completo
    n_samples : int
        Número de muestras para análisis (para acelerar)
    """
    print("\n" + "=" * 80)
    print("🔍 ANÁLISIS SHAP - RED 1: PREDICCIÓN DE POTENCIAL")
    print("=" * 80)
    
    model = model_data['model']
    features = model_data['features']
    
    # Preparar datos
    TARGET = "potential"
    df_clean = df[df[TARGET].notna()].copy()
    X = df_clean[features].to_numpy(dtype=float)
    y = df_clean[TARGET].to_numpy(dtype=float)
    
    # Tomar muestra para acelerar SHAP
    if len(X) > n_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X), size=n_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    print(f"\n📊 Usando {len(X_sample)} muestras para análisis SHAP")
    
    # Wrapper para el modelo (SHAP necesita una función callable)
    def model_predict(X_input):
        return model.predict(X_input)
    
    # Crear background dataset (100 muestras para kernel explainer)
    background_size = min(100, len(X_sample))
    rng = np.random.default_rng(42)
    background_indices = rng.choice(len(X_sample), size=background_size, replace=False)
    X_background = X_sample[background_indices]
    
    print(f"\n⏳ Calculando valores SHAP (esto puede tardar 2-5 minutos)...")
    
    # Crear explainer (KernelExplainer es agnóstico al modelo)
    explainer = shap.KernelExplainer(model_predict, X_background)
    
    # Calcular SHAP values para una muestra pequeña
    explain_size = min(200, len(X_sample))
    X_explain = X_sample[:explain_size]
    shap_values = explainer.shap_values(X_explain, nsamples=100)
    
    print("✅ Valores SHAP calculados")
    
    # =========================
    # IMPORTANCIA GLOBAL DE CARACTERÍSTICAS
    # =========================
    
    print("\n📊 Calculando importancia global de características...")
    
    # Importancia = promedio de |SHAP values|
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'feature': features,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    
    # Guardar
    importance_df.to_csv("shap_red1_feature_importance.csv", index=False)
    
    print("\n🏆 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES (SHAP):")
    print(importance_df.head(10).to_string(index=False))
    
    # =========================
    # VISUALIZACIONES SHAP
    # =========================
    
    print("\n📈 Generando visualizaciones SHAP...")
    
    # 1. Summary Plot (barras)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain, feature_names=features, 
                      plot_type="bar", show=False)
    plt.title("Red 1: Importancia Global de Características (SHAP)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("visualizations/shap_red1_summary_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ shap_red1_summary_bar.png")
    
    # 2. Summary Plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_explain, feature_names=features, show=False)
    plt.title("Red 1: Distribución de Valores SHAP", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("visualizations/shap_red1_summary_beeswarm.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ shap_red1_summary_beeswarm.png")
    
    # 3. Dependence plots para top 3 características
    top3_features = importance_df.head(3)['feature'].tolist()
    
    for i, feat in enumerate(top3_features):
        feat_idx = features.index(feat)
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(feat_idx, shap_values, X_explain, 
                             feature_names=features, show=False)
        plt.title(f"Red 1: Dependencia SHAP - {feat}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"visualizations/shap_red1_dependence_{feat}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ shap_red1_dependence_{feat}.png")
    
    # =========================
    # EXPLICACIONES LOCALES (EJEMPLOS)
    # =========================
    
    print("\n📋 Generando explicaciones locales para casos ejemplo...")
    
    # Seleccionar 5 casos: mejores y peores predicciones
    predictions = model.predict(X_explain)
    errors = np.abs(predictions - y_sample[:explain_size])
    
    best_indices = np.argsort(errors)[:3]  # 3 mejores
    worst_indices = np.argsort(errors)[-2:]  # 2 peores
    example_indices = list(best_indices) + list(worst_indices)
    
    local_explanations = []
    
    for idx in example_indices:
        local_exp = {
            'sample_id': int(idx),
            'true_value': float(y_sample[idx]),
            'predicted_value': float(predictions[idx]),
            'error': float(errors[idx]),
            'shap_contributions': {}
        }
        
        for feat_name, shap_val in zip(features, shap_values[idx]):
            local_exp['shap_contributions'][feat_name] = float(shap_val)
        
        # Ordenar por importancia
        local_exp['shap_contributions'] = dict(
            sorted(local_exp['shap_contributions'].items(), 
                   key=lambda x: abs(x[1]), reverse=True)
        )
        
        local_explanations.append(local_exp)
    
    # Guardar explicaciones locales
    with open("shap_red1_local_explanations.json", "w", encoding="utf-8") as f:
        json.dump(local_explanations, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ {len(local_explanations)} explicaciones locales guardadas")
    
    # =========================
    # RESUMEN
    # =========================
    
    results = {
        'n_samples_analyzed': int(explain_size),
        'n_features': len(features),
        'top_5_features': importance_df.head(5).to_dict('records'),
        'mean_shap_value': float(np.abs(shap_values).mean()),
        'max_shap_value': float(np.abs(shap_values).max())
    }
    
    with open("shap_red1_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Análisis SHAP Red 1 completado")
    
    return importance_df, shap_values


# =========================
# ANÁLISIS SHAP - RED 2 (CLASIFICACIÓN)
# =========================

def analyze_shap_red2(model_data, df, n_samples=1000):
    """
    Análisis SHAP para Red 2 (Clasificación de Perfil)
    """
    print("\n" + "=" * 80)
    print("🔍 ANÁLISIS SHAP - RED 2: CLASIFICACIÓN DE PERFIL")
    print("=" * 80)
    
    model = model_data['model']
    features = model_data['features']
    classes = model_data['classes']
    
    # Preparar datos
    y_position, _, _, _ = build_position_target_7_classes(df)
    X = df[features].to_numpy(dtype=float)
    
    # Balancear
    X_balanced, y_balanced = undersample_balance(X, y_position, seed=42)
    
    # Tomar muestra
    if len(X_balanced) > n_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_balanced), size=n_samples, replace=False)
        X_sample = X_balanced[indices]
        y_sample = y_balanced[indices]
    else:
        X_sample = X_balanced
        y_sample = y_balanced
    
    print(f"\n📊 Usando {len(X_sample)} muestras para análisis SHAP")
    print(f"📊 {len(classes)} clases: {', '.join(classes)}")
    
    # Wrapper para probabilidades
    def model_predict_proba(X_input):
        return model.predict_proba(X_input)
    
    # Background dataset
    background_size = min(100, len(X_sample))
    rng = np.random.default_rng(42)
    background_indices = rng.choice(len(X_sample), size=background_size, replace=False)
    X_background = X_sample[background_indices]
    
    print(f"\n⏳ Calculando valores SHAP para clasificación (2-5 minutos)...")
    
    # Crear explainer
    explainer = shap.KernelExplainer(model_predict_proba, X_background)
    
    # Calcular SHAP values
    explain_size = min(200, len(X_sample))
    X_explain = X_sample[:explain_size]
    shap_values = explainer.shap_values(X_explain, nsamples=100)
    
    print("✅ Valores SHAP calculados")
    
    # =========================
    # IMPORTANCIA POR CLASE
    # =========================
    
    print("\n📊 Calculando importancia de características por clase...")
    
    importance_results = []
    
    # shap_values es una lista de arrays (uno por clase)
    for class_idx, class_name in enumerate(classes):
        shap_class = shap_values[class_idx]
        importance = np.abs(shap_class).mean(axis=0)
        
        # CORRECCIÓN: Usar solo las features que coinciden con el tamaño de importance
        n_features_actual = min(len(features), len(importance))
        
        for feat_idx in range(n_features_actual):
            importance_results.append({
                'class': class_name,
                'feature': features[feat_idx],
                'shap_importance': float(importance[feat_idx])
            })
    
    importance_df = pd.DataFrame(importance_results)
    importance_df.to_csv("shap_red2_feature_importance_by_class.csv", index=False)
    
    # Importancia global (promedio sobre todas las clases)
    global_importance = importance_df.groupby('feature')['shap_importance'].mean().sort_values(ascending=False)
    
    print("\n🏆 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES (GLOBAL):")
    print(global_importance.head(10).to_string())
    
    # =========================
    # VISUALIZACIONES SHAP
    # =========================
    
    print("\n📈 Generando visualizaciones SHAP...")
    
    # 1. Summary plot para cada clase (top 3 clases)
    top_classes = classes[:3]
    
    for class_idx, class_name in enumerate(top_classes):
        if class_idx < len(shap_values):
            plt.figure(figsize=(10, 8))
            
            # CORRECCIÓN: Asegurar que X_explain y features coincidan con shap_values
            n_features_shap = shap_values[class_idx].shape[1]
            n_features_actual = min(len(features), n_features_shap)
            
            shap.summary_plot(shap_values[class_idx][:, :n_features_actual], 
                              X_explain[:, :n_features_actual], 
                              feature_names=features[:n_features_actual], 
                              plot_type="bar", show=False)
            plt.title(f"Red 2: Importancia para clase '{class_name}' (SHAP)", 
                      fontsize=14, fontweight='bold')
            plt.tight_layout()
            safe_name = class_name.replace(" ", "_").replace("/", "_")
            plt.savefig(f"visualizations/shap_red2_summary_{safe_name}.png", 
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ shap_red2_summary_{safe_name}.png")
    
    # 2. Importancia global (heatmap)
    plt.figure(figsize=(12, 8))
    pivot = importance_df.pivot(index='feature', columns='class', values='shap_importance')
    pivot = pivot.loc[global_importance.head(15).index]  # Top 15 features
    
    import matplotlib.cm as cm
    plt.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='SHAP Importance')
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=10)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha='right', fontsize=9)
    plt.title("Red 2: Importancia de Características por Clase (SHAP)", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("visualizations/shap_red2_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ shap_red2_heatmap.png")
    
    # =========================
    # RESUMEN
    # =========================
    
    results = {
        'n_samples_analyzed': int(explain_size),
        'n_features': len(features),
        'n_classes': len(classes),
        'classes': classes.tolist(),
        'top_5_features_global': global_importance.head(5).to_dict()
    }
    
    with open("shap_red2_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Análisis SHAP Red 2 completado")
    
    return importance_df, shap_values


# =========================
# MAIN
# =========================

def main():
    print("\n" + "=" * 80)
    print("🧠 ANÁLISIS DE INTERPRETABILIDAD CON SHAP")
    print("=" * 80)
    
    # Cargar modelos y datos
    red1_data, red2_data, df = load_models_and_data()
    
    if red1_data is None or red2_data is None or df is None:
        print("\n❌ No se pudieron cargar los modelos. Abortando.")
        return
    
    # Crear carpeta de visualizaciones si no existe
    Path("visualizations").mkdir(exist_ok=True)
    
    # Análisis Red 1
    imp_red1, shap_red1 = analyze_shap_red1(red1_data, df, n_samples=1000)
    
    # Análisis Red 2
    imp_red2, shap_red2 = analyze_shap_red2(red2_data, df, n_samples=1000)
    
    # Resumen final
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE ANÁLISIS SHAP")
    print("=" * 80)
    
    print("\n✅ ARCHIVOS GENERADOS:")
    print("\n🔹 RED 1 (REGRESIÓN):")
    print("   - shap_red1_feature_importance.csv")
    print("   - shap_red1_local_explanations.json")
    print("   - shap_red1_analysis_summary.json")
    print("   - visualizations/shap_red1_summary_bar.png")
    print("   - visualizations/shap_red1_summary_beeswarm.png")
    print("   - visualizations/shap_red1_dependence_*.png (top 3 features)")
    
    print("\n🔹 RED 2 (CLASIFICACIÓN):")
    print("   - shap_red2_feature_importance_by_class.csv")
    print("   - shap_red2_analysis_summary.json")
    print("   - visualizations/shap_red2_summary_*.png")
    print("   - visualizations/shap_red2_heatmap.png")
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS SHAP COMPLETADO")
    print("=" * 80)
    print("\n💡 Los valores SHAP muestran:")
    print("   • Qué características son más importantes")
    print("   • Cómo cada característica contribuye a las predicciones")
    print("   • Efectos de interacción entre características")
    print("   • Explicaciones individuales para cada predicción")


if __name__ == "__main__":
    main()
