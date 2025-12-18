"""
Comparativa con Modelos Baseline
=================================

Este script entrena modelos baseline simples y compara su rendimiento
con las redes neuronales MLP.

Modelos Baseline:
- Regresión: Regresión Lineal, Ridge, Decision Tree, Random Forest
- Clasificación: Regresión Logística, KNN, Decision Tree, Random Forest

Autor: Estudiante B
Fecha: Diciembre 2025
"""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import json
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    build_position_target_7_classes, undersample_balance,
    rmse, mae, ID_COLS
)


# =========================
# CARGAR DATOS Y MODELOS MLP
# =========================

def load_data_and_mlp_models():
    """
    Carga el dataset y los modelos MLP entrenados
    """
    print("=" * 80)
    print("📂 CARGANDO DATOS Y MODELOS MLP")
    print("=" * 80)
    
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
        raise FileNotFoundError("No se encuentra database_clean.sqlite")
    
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT * FROM player_attributes_clean;", conn)
    conn.close()
    
    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Cargar modelos MLP
    try:
        with open("red1_regresion_trained.pkl", "rb") as f:
            mlp_red1 = pickle.load(f)
        print("✅ MLP Red 1 (Regresión) cargada")
    except FileNotFoundError:
        print("⚠️  MLP Red 1 no encontrada - se omitirá en la comparativa")
        mlp_red1 = None
    
    try:
        with open("red2_clasificacion_trained.pkl", "rb") as f:
            mlp_red2 = pickle.load(f)
        print("✅ MLP Red 2 (Clasificación) cargada")
    except FileNotFoundError:
        print("⚠️  MLP Red 2 no encontrada - se omitirá en la comparativa")
        mlp_red2 = None
    
    # Cargar resultados MLP
    try:
        with open("estudiante_a_red1_results.json", "r", encoding="utf-8") as f:
            mlp_red1_results = json.load(f)
        print("✅ Resultados MLP Red 1 cargados")
    except FileNotFoundError:
        mlp_red1_results = None
    
    try:
        with open("estudiante_a_red2_results.json", "r", encoding="utf-8") as f:
            mlp_red2_results = json.load(f)
        print("✅ Resultados MLP Red 2 cargados")
    except FileNotFoundError:
        mlp_red2_results = None
    
    return df, mlp_red1, mlp_red2, mlp_red1_results, mlp_red2_results


# =========================
# DIVISIÓN DE DATOS (IGUAL QUE MLP)
# =========================

def stratified_train_test_split(X, y, test_size=0.15, random_state=42):
    """
    División estratificada (mismo método que MLP para comparar)
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    
    # Para regresión: división aleatoria
    if np.issubdtype(y.dtype, np.floating) and len(np.unique(y)) > 20:
        indices = np.arange(n)
        rng.shuffle(indices)
        
        n_test = int(n * test_size)
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    
    # Para clasificación: división estratificada
    classes = np.unique(y)
    train_idx, test_idx = [], []
    
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        
        n_cls = len(cls_idx)
        n_test_cls = int(n_cls * test_size)
        
        test_idx.extend(cls_idx[:n_test_cls])
        train_idx.extend(cls_idx[n_test_cls:])
    
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# =========================
# MODELOS BASELINE - REGRESIÓN
# =========================

def train_baseline_regression(df, features):
    """
    Entrena modelos baseline para regresión (predicción de potencial)
    
    Modelos:
    - Regresión Lineal
    - Ridge Regression
    - Decision Tree
    - Random Forest
    """
    print("\n" + "=" * 80)
    print("📊 ENTRENAMIENTO DE MODELOS BASELINE - REGRESIÓN")
    print("=" * 80)
    
    TARGET = "potential"
    df_clean = df[df[TARGET].notna()].copy()
    
    X = df_clean[features].to_numpy(dtype=float)
    y = df_clean[TARGET].to_numpy(dtype=float)
    
    # División de datos (85% train, 15% test como en MLP)
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    print(f"\n📂 División de datos:")
    print(f"   Train: {len(X_train):,} ejemplos")
    print(f"   Test:  {len(X_test):,} ejemplos")
    
    # Definir modelos baseline
    models = {
        'Linear_Regression': LinearRegression(),
        'Ridge_Regression': Ridge(alpha=1.0),
        'Decision_Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random_Forest': RandomForestRegressor(n_estimators=100, max_depth=10, 
                                               random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_models = {}
    
    print("\n🏋️ Entrenando modelos baseline...\n")
    
    for name, model in models.items():
        print(f"   Entrenando {name}...", end=" ")
        start_time = time.time()
        
        # Entrenar
        model.fit(X_train, y_train)
        
        # Predecir
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Métricas
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        elapsed = time.time() - start_time
        
        result = {
            'model': name,
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'training_time_seconds': float(elapsed)
        }
        
        results.append(result)
        trained_models[name] = model
        
        print(f"✅ ({elapsed:.1f}s) | Test RMSE: {test_rmse:.3f} | R²: {test_r2:.4f}")
    
    # Guardar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv("baseline_regression_results.csv", index=False)
    
    print("\n✅ Modelos baseline de regresión entrenados")
    print(f"   Resultados guardados en: baseline_regression_results.csv")
    
    return results_df, trained_models, X_test, y_test


# =========================
# MODELOS BASELINE - CLASIFICACIÓN
# =========================

def train_baseline_classification(df, features, classes):
    """
    Entrena modelos baseline para clasificación (perfil de jugador)
    
    Modelos:
    - Regresión Logística
    - K-Nearest Neighbors
    - Decision Tree
    - Random Forest
    """
    print("\n" + "=" * 80)
    print("📊 ENTRENAMIENTO DE MODELOS BASELINE - CLASIFICACIÓN")
    print("=" * 80)
    
    # Preparar datos
    y_position, _, _, _ = build_position_target_7_classes(df)
    X = df[features].to_numpy(dtype=float)
    
    # Balancear clases
    X_balanced, y_balanced = undersample_balance(X, y_position, seed=42)
    
    print(f"\n📊 Datos balanceados: {len(X_balanced)} ejemplos")
    print(f"📊 {len(classes)} clases: {', '.join(classes)}")
    
    # División de datos
    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X_balanced, y_balanced, test_size=0.15, random_state=42
    )
    
    print(f"\n📂 División de datos:")
    print(f"   Train: {len(X_train):,} ejemplos")
    print(f"   Test:  {len(X_test):,} ejemplos")
    
    # Definir modelos baseline
    models = {
        'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        'Decision_Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                random_state=42, n_jobs=-1)
    }
    
    results = []
    trained_models = {}
    
    print("\n🏋️ Entrenando modelos baseline...\n")
    
    for name, model in models.items():
        print(f"   Entrenando {name}...", end=" ")
        start_time = time.time()
        
        # Entrenar
        model.fit(X_train, y_train)
        
        # Predecir
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Métricas
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        # Precision, Recall, F1 (macro average)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test, test_pred, average='macro', zero_division=0
        )
        
        elapsed = time.time() - start_time
        
        result = {
            'model': name,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'training_time_seconds': float(elapsed)
        }
        
        results.append(result)
        trained_models[name] = model
        
        print(f"✅ ({elapsed:.1f}s) | Test Acc: {test_acc:.3f} | F1: {test_f1:.3f}")
    
    # Guardar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv("baseline_classification_results.csv", index=False)
    
    print("\n✅ Modelos baseline de clasificación entrenados")
    print(f"   Resultados guardados en: baseline_classification_results.csv")
    
    return results_df, trained_models, X_test, y_test


# =========================
# COMPARATIVA MLP vs BASELINE
# =========================

def compare_with_mlp_regression(baseline_results, mlp_results):
    """
    Compara resultados de modelos baseline con MLP (regresión)
    """
    print("\n" + "=" * 80)
    print("📊 COMPARATIVA: MLP vs BASELINE - REGRESIÓN")
    print("=" * 80)
    
    # Agregar MLP a la tabla
    if mlp_results:
        mlp_row = {
            'model': 'MLP_Neural_Network',
            'train_rmse': mlp_results['results']['train']['RMSE'],
            'test_rmse': mlp_results['results']['test']['RMSE'],
            'train_mae': mlp_results['results']['train']['MAE'],
            'test_mae': mlp_results['results']['test']['MAE'],
            'train_r2': mlp_results['results']['train']['R2'],
            'test_r2': mlp_results['results']['test']['R2'],
            'training_time_seconds': None  # No registrado
        }
        
        comparison = pd.concat([
            baseline_results,
            pd.DataFrame([mlp_row])
        ], ignore_index=True)
    else:
        comparison = baseline_results.copy()
    
    # Ordenar por Test RMSE (menor es mejor)
    comparison = comparison.sort_values('test_rmse')
    
    print("\n🏆 RANKING DE MODELOS (por Test RMSE - menor es mejor):\n")
    print(comparison[['model', 'test_rmse', 'test_mae', 'test_r2', 'training_time_seconds']].to_string(index=False))
    
    # Guardar comparativa
    comparison.to_csv("comparativa_regresion_completa.csv", index=False)
    
    # Análisis
    best_model = comparison.iloc[0]
    print(f"\n🥇 MEJOR MODELO: {best_model['model']}")
    print(f"   Test RMSE: {best_model['test_rmse']:.3f}")
    print(f"   Test R²:   {best_model['test_r2']:.4f}")
    
    if mlp_results:
        mlp_rank = comparison[comparison['model'] == 'MLP_Neural_Network'].index[0] + 1
        total_models = len(comparison)
        print(f"\n📍 MLP Neural Network: Posición {mlp_rank}/{total_models}")
        
        if mlp_rank == 1:
            print("   ✅ MLP es el MEJOR modelo")
        elif mlp_rank <= 2:
            print("   ✅ MLP está entre los TOP 2 modelos")
        else:
            best_baseline = comparison[comparison['model'] != 'MLP_Neural_Network'].iloc[0]
            improvement = ((mlp_results['results']['test']['RMSE'] - best_baseline['test_rmse']) / 
                          best_baseline['test_rmse'] * 100)
            print(f"   ⚠️  Mejor baseline ({best_baseline['model']}) supera a MLP en {abs(improvement):.1f}%")
    
    return comparison


def compare_with_mlp_classification(baseline_results, mlp_results):
    """
    Compara resultados de modelos baseline con MLP (clasificación)
    """
    print("\n" + "=" * 80)
    print("📊 COMPARATIVA: MLP vs BASELINE - CLASIFICACIÓN")
    print("=" * 80)
    
    # Agregar MLP a la tabla
    if mlp_results:
        mlp_row = {
            'model': 'MLP_Neural_Network',
            'train_accuracy': mlp_results['results']['train_accuracy'],
            'test_accuracy': mlp_results['results']['test_accuracy'],
            'test_precision': None,  # No calculado en formato simple
            'test_recall': None,
            'test_f1': None,
            'training_time_seconds': None
        }
        
        comparison = pd.concat([
            baseline_results,
            pd.DataFrame([mlp_row])
        ], ignore_index=True)
    else:
        comparison = baseline_results.copy()
    
    # Ordenar por Test Accuracy (mayor es mejor)
    comparison = comparison.sort_values('test_accuracy', ascending=False)
    
    print("\n🏆 RANKING DE MODELOS (por Test Accuracy - mayor es mejor):\n")
    print(comparison[['model', 'test_accuracy', 'test_f1', 'training_time_seconds']].to_string(index=False))
    
    # Guardar comparativa
    comparison.to_csv("comparativa_clasificacion_completa.csv", index=False)
    
    # Análisis
    best_model = comparison.iloc[0]
    print(f"\n🥇 MEJOR MODELO: {best_model['model']}")
    print(f"   Test Accuracy: {best_model['test_accuracy']:.3f}")
    
    if mlp_results:
        mlp_rank = comparison[comparison['model'] == 'MLP_Neural_Network'].index[0] + 1
        total_models = len(comparison)
        print(f"\n📍 MLP Neural Network: Posición {mlp_rank}/{total_models}")
        
        if mlp_rank == 1:
            print("   ✅ MLP es el MEJOR modelo")
        elif mlp_rank <= 2:
            print("   ✅ MLP está entre los TOP 2 modelos")
        else:
            best_baseline = comparison[comparison['model'] != 'MLP_Neural_Network'].iloc[0]
            improvement = ((mlp_results['results']['test_accuracy'] - best_baseline['test_accuracy']) / 
                          best_baseline['test_accuracy'] * 100)
            print(f"   ⚠️  Mejor baseline ({best_baseline['model']}) supera a MLP en {abs(improvement):.1f}%")
    
    return comparison


# =========================
# VISUALIZACIONES
# =========================

def create_comparison_visualizations(comp_reg, comp_clf):
    """
    Crea gráficos comparativos
    """
    print("\n" + "=" * 80)
    print("📈 GENERANDO VISUALIZACIONES COMPARATIVAS")
    print("=" * 80)
    
    Path("visualizations").mkdir(exist_ok=True)
    
    # 1. Comparativa Regresión - Test RMSE
    plt.figure(figsize=(10, 6))
    models = comp_reg['model'].values
    rmse_values = comp_reg['test_rmse'].values
    
    colors = ['#FF6B6B' if m == 'MLP_Neural_Network' else '#4ECDC4' for m in models]
    
    bars = plt.barh(models, rmse_values, color=colors)
    plt.xlabel('Test RMSE (menor es mejor)', fontsize=12, fontweight='bold')
    plt.title('Comparativa de Modelos - Regresión (Predicción de Potencial)', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Añadir valores
    for i, (model, val) in enumerate(zip(models, rmse_values)):
        plt.text(val + 0.05, i, f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("visualizations/comparativa_regresion.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ comparativa_regresion.png")
    
    # 2. Comparativa Clasificación - Test Accuracy
    plt.figure(figsize=(10, 6))
    models = comp_clf['model'].values
    acc_values = comp_clf['test_accuracy'].values
    
    colors = ['#FF6B6B' if m == 'MLP_Neural_Network' else '#95E1D3' for m in models]
    
    bars = plt.barh(models, acc_values, color=colors)
    plt.xlabel('Test Accuracy (mayor es mejor)', fontsize=12, fontweight='bold')
    plt.title('Comparativa de Modelos - Clasificación (Perfil de Jugador)', 
              fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', alpha=0.3)
    
    # Añadir valores
    for i, (model, val) in enumerate(zip(models, acc_values)):
        plt.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig("visualizations/comparativa_clasificacion.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ comparativa_clasificacion.png")
    
    # 3. Comparativa de tiempos de entrenamiento (Regresión)
    if comp_reg['training_time_seconds'].notna().any():
        plt.figure(figsize=(10, 6))
        comp_reg_time = comp_reg[comp_reg['training_time_seconds'].notna()].copy()
        
        models = comp_reg_time['model'].values
        times = comp_reg_time['training_time_seconds'].values
        
        plt.barh(models, times, color='#F38181')
        plt.xlabel('Tiempo de Entrenamiento (segundos)', fontsize=12, fontweight='bold')
        plt.title('Comparativa de Tiempos de Entrenamiento - Regresión', 
                  fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        for i, (model, val) in enumerate(zip(models, times)):
            plt.text(val + 1, i, f'{val:.1f}s', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig("visualizations/comparativa_tiempos_regresion.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ comparativa_tiempos_regresion.png")
    
    print("\n✅ Visualizaciones guardadas en carpeta 'visualizations/'")


# =========================
# MAIN
# =========================

def main():
    print("=" * 80)
    print("🔬 COMPARATIVA DE MODELOS: MLP vs BASELINE")
    print("=" * 80)
    
    # Cargar datos
    df, mlp_red1, mlp_red2, mlp_red1_results, mlp_red2_results = load_data_and_mlp_models()
    
    # ==========================================
    # REGRESIÓN
    # ==========================================
    
    if mlp_red1 and mlp_red1_results:
        features_red1 = mlp_red1['features']
        
        # Entrenar baseline
        baseline_reg_results, baseline_reg_models, X_test_reg, y_test_reg = train_baseline_regression(
            df, features_red1
        )
        
        # Comparar
        comparison_reg = compare_with_mlp_regression(baseline_reg_results, mlp_red1_results)
    else:
        print("\n⚠️  MLP Red 1 no disponible - omitiendo comparativa de regresión")
        comparison_reg = None
    
    # ==========================================
    # CLASIFICACIÓN
    # ==========================================
    
    if mlp_red2 and mlp_red2_results:
        features_red2 = mlp_red2['features']
        classes_red2 = mlp_red2['classes']
        
        # Entrenar baseline
        baseline_clf_results, baseline_clf_models, X_test_clf, y_test_clf = train_baseline_classification(
            df, features_red2, classes_red2
        )
        
        # Comparar
        comparison_clf = compare_with_mlp_classification(baseline_clf_results, mlp_red2_results)
    else:
        print("\n⚠️  MLP Red 2 no disponible - omitiendo comparativa de clasificación")
        comparison_clf = None
    
    # ==========================================
    # VISUALIZACIONES
    # ==========================================
    
    if comparison_reg is not None and comparison_clf is not None:
        create_comparison_visualizations(comparison_reg, comparison_clf)
    
    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE COMPARATIVA")
    print("=" * 80)
    
    print("\n✅ ARCHIVOS GENERADOS:")
    if comparison_reg is not None:
        print("\n🔹 REGRESIÓN:")
        print("   - baseline_regression_results.csv")
        print("   - comparativa_regresion_completa.csv")
        print("   - visualizations/comparativa_regresion.png")
    
    if comparison_clf is not None:
        print("\n🔹 CLASIFICACIÓN:")
        print("   - baseline_classification_results.csv")
        print("   - comparativa_clasificacion_completa.csv")
        print("   - visualizations/comparativa_clasificacion.png")
    
    print("\n" + "=" * 80)
    print("✅ COMPARATIVA COMPLETADA")
    print("=" * 80)
    
    print("\n💡 CONCLUSIONES:")
    print("   • Los modelos baseline proporcionan una referencia de rendimiento")
    print("   • Las redes neuronales MLP pueden capturar patrones no lineales complejos")
    print("   • La comparativa justifica el uso de arquitecturas más sofisticadas")
    print("   • El tiempo de entrenamiento es un factor a considerar en producción")


if __name__ == "__main__":
    main()
