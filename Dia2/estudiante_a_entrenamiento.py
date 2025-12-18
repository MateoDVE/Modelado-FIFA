"""
Estudiante A: Entrenamiento y Optimización
==========================================

Responsabilidades:
1. División de datos: 70% train, 15% val, 15% test
2. Balanceo y estratificación por clase
3. Búsqueda de hiperparámetros óptimos
4. Entrenamiento con validación

Autor: Estudiante A
Fecha: Diciembre 2025
"""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import json
import pickle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import (
    MLPRegressor, MLPClassifier,
    build_position_target_7_classes, undersample_balance,
    rmse, mae, r2_score, confusion_matrix, ID_COLS
)


# =========================
# DIVISIÓN ESTRATIFICADA DE DATOS
# =========================

def stratified_train_val_test_split(X, y, train_size=0.70, val_size=0.15, 
                                     test_size=0.15, random_state=42):
    """
    División estratificada de datos en train/val/test.
    
    Mantiene la proporción de clases en cada conjunto.
    
    Parámetros:
    -----------
    X : np.ndarray
        Características
    y : np.ndarray
        Etiquetas (para clasificación) o valores (para regresión)
    train_size : float
        Proporción para entrenamiento (default: 0.70)
    val_size : float
        Proporción para validación (default: 0.15)
    test_size : float
        Proporción para test (default: 0.15)
    random_state : int
        Semilla aleatoria
    
    Retorna:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Las proporciones deben sumar 1.0"
    
    rng = np.random.default_rng(random_state)
    n = len(y)
    
    # Para regresión: dividir sin estratificación
    if np.issubdtype(y.dtype, np.floating) and len(np.unique(y)) > 20:
        indices = np.arange(n)
        rng.shuffle(indices)
        
        n_train = int(n * train_size)
        n_val = int(n * val_size)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        return (X[train_idx], X[val_idx], X[test_idx],
                y[train_idx], y[val_idx], y[test_idx])
    
    # Para clasificación: división estratificada
    classes = np.unique(y)
    train_idx, val_idx, test_idx = [], [], []
    
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        
        n_cls = len(cls_idx)
        n_train_cls = int(n_cls * train_size)
        n_val_cls = int(n_cls * val_size)
        
        train_idx.extend(cls_idx[:n_train_cls])
        val_idx.extend(cls_idx[n_train_cls:n_train_cls + n_val_cls])
        test_idx.extend(cls_idx[n_train_cls + n_val_cls:])
    
    # Mezclar índices
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)
    
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    
    return (X[train_idx], X[val_idx], X[test_idx],
            y[train_idx], y[val_idx], y[test_idx])


# =========================
# BÚSQUEDA DE HIPERPARÁMETROS - RED 1 (REGRESIÓN)
# ===================python estudiante_a_entrenamiento.py======

def hyperparameter_search_regression(X_train, y_train, X_val, y_val, verbose=True):
    """
    Búsqueda de hiperparámetros óptimos para Red 1 (Regresión).
    
    Hiperparámetros a probar:
    - learning_rate: [0.0001, 0.001, 0.01]
    - l2_lambda: [0.001, 0.01, 0.1]
    - n_iter: [2000, 3000, 5000]
    
    Retorna:
    --------
    dict con mejores hiperparámetros y resultados
    """
    
    if verbose:
        print("\n" + "="*80)
        print("🔍 BÚSQUEDA DE HIPERPARÁMETROS - RED 1 (REGRESIÓN)")
        print("="*80)
    
    # Grid de hiperparámetros (ESTABLE - Sin overflow)
    param_grid = {
        'learning_rate': [0.0003, 0.0005],  # Solo valores estables
        'l2_lambda': [0.01, 0.05],
        'n_iter': [2000, 2500]  # Reducido para evitar overflow
    }
    
    best_score = float('inf')  # Queremos minimizar RMSE
    best_params = None
    results = []
    
    total_combinations = (len(param_grid['learning_rate']) * 
                         len(param_grid['l2_lambda']) * 
                         len(param_grid['n_iter']))
    
    if verbose:
        print(f"🔢 Probando {total_combinations} combinaciones de hiperparámetros...\n")
    
    iteration = 0
    for lr in param_grid['learning_rate']:
        for l2 in param_grid['l2_lambda']:
            for n_iter in param_grid['n_iter']:
                iteration += 1
                
                # Entrenar modelo
                mlp = MLPRegressor(
                    layer_sizes=[20, 256, 128, 64, 1],
                    learning_rate=lr,
                    n_iter=n_iter,
                    l2_lambda=l2,
                    verbose=False
                )
                
                mlp.fit(X_train, y_train)
                
                # Evaluar en validación
                val_pred = mlp.predict(X_val)
                val_rmse = rmse(y_val, val_pred)
                val_r2 = r2_score(y_val, val_pred)
                
                # Evaluar en entrenamiento (para detectar overfitting)
                train_pred = mlp.predict(X_train)
                train_rmse = rmse(y_train, train_pred)
                
                result = {
                    'learning_rate': lr,
                    'l2_lambda': l2,
                    'n_iter': n_iter,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'train_rmse': train_rmse,
                    'gap': train_rmse - val_rmse
                }
                results.append(result)
                
                if verbose:
                    print(f"[{iteration}/{total_combinations}] "
                          f"lr={lr:.4f}, λ={l2:.3f}, iter={n_iter} | "
                          f"Val RMSE={val_rmse:.3f}, R²={val_r2:.4f}, "
                          f"Gap={result['gap']:.3f}")
                
                # Actualizar mejor
                if val_rmse < best_score:
                    best_score = val_rmse
                    best_params = {
                        'learning_rate': lr,
                        'l2_lambda': l2,
                        'n_iter': n_iter
                    }
    
    if verbose:
        print("\n" + "="*80)
        print("🏆 MEJORES HIPERPARÁMETROS ENCONTRADOS:")
        print(f"   Learning Rate: {best_params['learning_rate']}")
        print(f"   L2 Lambda:     {best_params['l2_lambda']}")
        print(f"   Iteraciones:   {best_params['n_iter']}")
        print(f"   Val RMSE:      {best_score:.3f}")
        print("="*80)
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }


# =========================
# BÚSQUEDA DE HIPERPARÁMETROS - RED 2 (CLASIFICACIÓN)
# =========================

def hyperparameter_search_classification(X_train, y_train, X_val, y_val, 
                                          n_classes, verbose=True):
    """
    Búsqueda de hiperparámetros óptimos para Red 2 (Clasificación).
    
    Retorna:
    --------
    dict con mejores hiperparámetros y resultados
    """
    
    if verbose:
        print("\n" + "="*80)
        print("🔍 BÚSQUEDA DE HIPERPARÁMETROS - RED 2 (CLASIFICACIÓN)")
        print("="*80)
    
    # Grid de hiperparámetros (ESTABLE - Sin overflow)
    param_grid = {
        'learning_rate': [0.0003, 0.0005],  # Solo valores estables
        'l2_lambda': [0.01, 0.05],
        'n_iter': [2000, 2500]  # Reducido para evitar overflow
    }
    
    best_score = 0.0  # Queremos maximizar accuracy
    best_params = None
    results = []
    
    total_combinations = (len(param_grid['learning_rate']) * 
                         len(param_grid['l2_lambda']) * 
                         len(param_grid['n_iter']))
    
    if verbose:
        print(f"🔢 Probando {total_combinations} combinaciones de hiperparámetros...\n")
    
    iteration = 0
    for lr in param_grid['learning_rate']:
        for l2 in param_grid['l2_lambda']:
            for n_iter in param_grid['n_iter']:
                iteration += 1
                
                # Entrenar modelo
                mlp = MLPClassifier(
                    layer_sizes=[15, 256, 128, n_classes],
                    learning_rate=lr,
                    n_iter=n_iter,
                    l2_lambda=l2,
                    verbose=False
                )
                
                mlp.fit(X_train, y_train)
                
                # Evaluar en validación
                val_acc = mlp.accuracy(X_val, y_val)
                
                # Evaluar en entrenamiento
                train_acc = mlp.accuracy(X_train, y_train)
                
                result = {
                    'learning_rate': lr,
                    'l2_lambda': l2,
                    'n_iter': n_iter,
                    'val_accuracy': val_acc,
                    'train_accuracy': train_acc,
                    'gap': train_acc - val_acc
                }
                results.append(result)
                
                if verbose:
                    print(f"[{iteration}/{total_combinations}] "
                          f"lr={lr:.4f}, λ={l2:.3f}, iter={n_iter} | "
                          f"Val Acc={val_acc:.3f}, Train Acc={train_acc:.3f}, "
                          f"Gap={result['gap']:.3f}")
                
                # Actualizar mejor
                if val_acc > best_score:
                    best_score = val_acc
                    best_params = {
                        'learning_rate': lr,
                        'l2_lambda': l2,
                        'n_iter': n_iter
                    }
    
    if verbose:
        print("\n" + "="*80)
        print("🏆 MEJORES HIPERPARÁMETROS ENCONTRADOS:")
        print(f"   Learning Rate: {best_params['learning_rate']}")
        print(f"   L2 Lambda:     {best_params['l2_lambda']}")
        print(f"   Iteraciones:   {best_params['n_iter']}")
        print(f"   Val Accuracy:  {best_score:.3f}")
        print("="*80)
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results
    }


# =========================
# ENTRENAMIENTO COMPLETO - RED 1
# =========================

def train_red1_with_optimization(df, verbose=True):
    """
    Entrenamiento completo de Red 1 con optimización de hiperparámetros.
    
    Proceso:
    1. Selección de características
    2. División estratificada (70/15/15)
    3. Búsqueda de hiperparámetros
    4. Entrenamiento con mejores parámetros
    5. Evaluación en test
    """
    
    if verbose:
        print("\n" + "="*80)
        print("🚀 ESTUDIANTE A: ENTRENAMIENTO RED 1 - PREDICCIÓN DE POTENCIAL")
        print("="*80)
    
    TARGET = "potential"
    
    # Verificar columna objetivo
    if TARGET not in df.columns:
        raise ValueError(f"Columna '{TARGET}' no encontrada")
    
    df_clean = df[df[TARGET].notna()].copy()
    
    if verbose:
        print(f"\n📊 Dataset: {len(df_clean)} jugadores con '{TARGET}' definido")
    
    # Seleccionar top 20 características
    num_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]
    candidates = [c for c in num_cols 
                  if c not in ID_COLS and c != TARGET and c != 'overall_rating']
    
    corrs = df_clean[candidates].corrwith(df_clean[TARGET]).abs().sort_values(ascending=False)
    top20_features = corrs.head(20).index.tolist()
    
    if verbose:
        print(f"\n✅ Top 20 características seleccionadas")
        print(f"   Correlación promedio con '{TARGET}': {corrs.head(20).mean():.3f}")
    
    X = df_clean[top20_features].to_numpy(dtype=float)
    y = df_clean[TARGET].to_numpy(dtype=float)
    
    # 📊 ENTRENAMIENTO COMPLETO: Usar todos los datos
    # Para prueba rápida, descomentar las líneas siguientes:
    # if len(X) > 20000:
    #     rng = np.random.default_rng(42)
    #     indices = rng.choice(len(X), size=20000, replace=False)
    #     X = X[indices]
    #     y = y[indices]
    #     if verbose:
    #         print(f"\n⚡ MODO PRUEBA: Usando subset de 20,000 ejemplos")
    
    if verbose:
        print(f"\n📊 Usando dataset completo: {len(X):,} ejemplos")
    
    # División estratificada
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X, y, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42
    )
    
    if verbose:
        print(f"\n📂 División de datos:")
        print(f"   Train: {len(X_train):5d} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Val:   {len(X_val):5d} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test:  {len(X_test):5d} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Búsqueda de hiperparámetros
    hp_results = hyperparameter_search_regression(
        X_train, y_train, X_val, y_val, verbose=verbose
    )
    
    best_params = hp_results['best_params']
    
    # Entrenar modelo final con mejores parámetros
    if verbose:
        print(f"\n🎯 Entrenando modelo final con mejores hiperparámetros...")
    
    final_model = MLPRegressor(
        layer_sizes=[20, 256, 128, 64, 1],
        learning_rate=best_params['learning_rate'],
        n_iter=best_params['n_iter'],
        l2_lambda=best_params['l2_lambda'],
        verbose=verbose
    )
    
    final_model.fit(X_train, y_train)
    
    # Evaluación en todos los conjuntos
    train_pred = final_model.predict(X_train)
    val_pred = final_model.predict(X_val)
    test_pred = final_model.predict(X_test)
    
    results = {
        'train': {
            'RMSE': rmse(y_train, train_pred),
            'MAE': mae(y_train, train_pred),
            'R2': r2_score(y_train, train_pred)
        },
        'val': {
            'RMSE': rmse(y_val, val_pred),
            'MAE': mae(y_val, val_pred),
            'R2': r2_score(y_val, val_pred)
        },
        'test': {
            'RMSE': rmse(y_test, test_pred),
            'MAE': mae(y_test, test_pred),
            'R2': r2_score(y_test, test_pred)
        }
    }
    
    if verbose:
        print("\n" + "="*80)
        print("📊 RESULTADOS FINALES - RED 1")
        print("="*80)
        for split_name, metrics in results.items():
            print(f"\n{split_name.upper():5s}:")
            print(f"  RMSE: {metrics['RMSE']:.3f}")
            print(f"  MAE:  {metrics['MAE']:.3f}")
            print(f"  R²:   {metrics['R2']:.4f}")
        
        # Diagnóstico
        gap = results['train']['RMSE'] - results['test']['RMSE']
        print(f"\n📈 Diagnóstico:")
        print(f"  Gap Train-Test: {gap:.3f}")
        if abs(gap) < 1.0:
            print("  ✅ Buen balance bias-variance")
        elif gap < -1.0:
            print("  ⚠️  Posible underfitting")
        else:
            print("  ⚠️  Posible overfitting (pero regularización ayuda)")
    
    return {
        'model': final_model,
        'features': top20_features,
        'best_params': best_params,
        'results': results,
        'hp_search': hp_results['all_results']
    }


# =========================
# ENTRENAMIENTO COMPLETO - RED 2
# =========================

def train_red2_with_optimization(df, verbose=True):
    """
    Entrenamiento completo de Red 2 con optimización de hiperparámetros.
    """
    
    if verbose:
        print("\n" + "="*80)
        print("🚀 ESTUDIANTE A: ENTRENAMIENTO RED 2 - CLASIFICACIÓN DE PERFIL")
        print("="*80)
    
    # Crear target por reglas (7 posiciones específicas)
    y_position, _, _, _ = build_position_target_7_classes(df)
    
    # Seleccionar 15 características clave
    key_features = [
        'reactions', 'ball_control', 'dribbling', 'short_passing', 'long_passing',
        'acceleration', 'sprint_speed', 'stamina', 'strength', 'positioning',
        'finishing', 'shot_power', 'marking', 'standing_tackle', 'interceptions'
    ]
    
    available_features = [f for f in key_features if f in df.columns]
    if len(available_features) < 15:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        candidates = [c for c in num_cols if c not in ID_COLS and c not in available_features]
        needed = 15 - len(available_features)
        available_features.extend(candidates[:needed])
    
    features_15 = available_features[:15]
    
    if verbose:
        print(f"\n✅ 15 características seleccionadas")
    
    X = df[features_15].to_numpy(dtype=float)
    
    # Balancear clases
    X_balanced, y_balanced = undersample_balance(X, y_position, seed=42)
    
    classes = np.unique(y_balanced)
    n_classes = len(classes)
    
    if verbose:
        print(f"\n📊 Clases balanceadas: {n_classes} posiciones")
        for cls in classes:
            count = np.sum(y_balanced == cls)
            print(f"   {cls:15s}: {count:4d} ejemplos")
    
    # División estratificada
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X_balanced, y_balanced, train_size=0.70, val_size=0.15, test_size=0.15, 
        random_state=42
    )
    
    if verbose:
        print(f"\n📂 División de datos:")
        print(f"   Train: {len(X_train):5d} ({len(X_train)/len(X_balanced)*100:.1f}%)")
        print(f"   Val:   {len(X_val):5d} ({len(X_val)/len(X_balanced)*100:.1f}%)")
        print(f"   Test:  {len(X_test):5d} ({len(X_test)/len(X_balanced)*100:.1f}%)")
    
    # Búsqueda de hiperparámetros
    hp_results = hyperparameter_search_classification(
        X_train, y_train, X_val, y_val, n_classes, verbose=verbose
    )
    
    best_params = hp_results['best_params']
    
    # Entrenar modelo final
    if verbose:
        print(f"\n🎯 Entrenando modelo final con mejores hiperparámetros...")
    
    final_model = MLPClassifier(
        layer_sizes=[15, 256, 128, n_classes],
        learning_rate=best_params['learning_rate'],
        n_iter=best_params['n_iter'],
        l2_lambda=best_params['l2_lambda'],
        verbose=verbose
    )
    
    final_model.fit(X_train, y_train)
    
    # Evaluación
    train_acc = final_model.accuracy(X_train, y_train)
    val_acc = final_model.accuracy(X_val, y_val)
    test_acc = final_model.accuracy(X_test, y_test)
    
    # Matriz de confusión en test
    cm_test = final_model.confusion_matrix(X_test, y_test)
    
    results = {
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'confusion_matrix': cm_test
    }
    
    if verbose:
        print("\n" + "="*80)
        print("📊 RESULTADOS FINALES - RED 2")
        print("="*80)
        print(f"\nTRAIN Accuracy: {train_acc:.3f}")
        print(f"VAL   Accuracy: {val_acc:.3f}")
        print(f"TEST  Accuracy: {test_acc:.3f}")
        
        print(f"\n📊 Matriz de Confusión (Test):")
        print(f"Clases: {list(classes)}")
        print(cm_test)
        
        gap = train_acc - test_acc
        print(f"\n📈 Diagnóstico:")
        print(f"  Gap Train-Test: {gap:.3f}")
        if abs(gap) < 0.05:
            print("  ✅ Excelente generalización")
        elif abs(gap) < 0.10:
            print("  ✅ Buena generalización")
        else:
            print("  ⚠️  Verificar overfitting")
    
    return {
        'model': final_model,
        'features': features_15,
        'classes': classes,
        'best_params': best_params,
        'results': results,
        'hp_search': hp_results['all_results']
    }


# =========================
# MAIN
# =========================

def main():
    """
    Función principal para ejecutar todo el proceso del Estudiante A
    """
    
    print("="*80)
    print("ESTUDIANTE A: ENTRENAMIENTO Y OPTIMIZACIÓN")
    print("="*80)
    
    # Cargar datos
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
            print(f"\n✓ Base de datos encontrada: {DB}")
            break
    
    if DB is None:
        raise FileNotFoundError("No se encuentra database_clean.sqlite")
    
    TABLE = "player_attributes_clean"
    
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE};", conn)
    conn.close()
    
    print(f"✓ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # ==========================================
    # RED 1: PREDICCIÓN DE POTENCIAL
    # ==========================================
    results_red1 = train_red1_with_optimization(df, verbose=True)
    
    # Guardar resultados
    pd.DataFrame(results_red1['hp_search']).to_csv(
        "estudiante_a_red1_hyperparameters.csv", index=False
    )
    
    with open("estudiante_a_red1_results.json", "w", encoding="utf-8") as f:
        json.dump({
            'architecture': [20, 256, 128, 64, 1],
            'features': results_red1['features'],
            'best_params': results_red1['best_params'],
            'results': results_red1['results']
        }, f, indent=2, ensure_ascii=False)
    
    # Guardar modelo entrenado
    with open("red1_regresion_trained.pkl", "wb") as f:
        pickle.dump({
            'model': results_red1['model'],
            'features': results_red1['features'],
            'best_params': results_red1['best_params']
        }, f)
    
    print("\n✅ Red 1: Resultados guardados en:")
    print("   - estudiante_a_red1_hyperparameters.csv")
    print("   - estudiante_a_red1_results.json")
    print("   - red1_regresion_trained.pkl (MODELO GUARDADO)")
    
    # ==========================================
    # RED 2: CLASIFICACIÓN DE PERFIL
    # ==========================================
    results_red2 = train_red2_with_optimization(df, verbose=True)
    
    # Guardar resultados
    pd.DataFrame(results_red2['hp_search']).to_csv(
        "estudiante_a_red2_hyperparameters.csv", index=False
    )
    
    pd.DataFrame(
        results_red2['results']['confusion_matrix'],
        index=[f"real_{c}" for c in results_red2['classes']],
        columns=[f"pred_{c}" for c in results_red2['classes']]
    ).to_csv("estudiante_a_red2_confusion_matrix.csv")
    
    # Guardar modelo entrenado
    with open("red2_clasificacion_trained.pkl", "wb") as f:
        pickle.dump({
            'model': results_red2['model'],
            'features': results_red2['features'],
            'classes': results_red2['classes'],
            'best_params': results_red2['best_params']
        }, f)
    
    with open("estudiante_a_red2_results.json", "w", encoding="utf-8") as f:
        json.dump({
            'architecture': [15, 256, 128, len(results_red2['classes'])],
            'features': results_red2['features'],
            'classes': results_red2['classes'].tolist(),
            'best_params': results_red2['best_params'],
            'results': {
                'train_accuracy': float(results_red2['results']['train_accuracy']),
                'val_accuracy': float(results_red2['results']['val_accuracy']),
                'test_accuracy': float(results_red2['results']['test_accuracy'])
            }
        }, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Red 2: Resultados guardados en:")
    print("   - estudiante_a_red2_hyperparameters.csv")
    print("   - estudiante_a_red2_confusion_matrix.csv")
    print("   - estudiante_a_red2_results.json")
    print("   - red2_clasificacion_trained.pkl (MODELO GUARDADO)")
    
    # ==========================================
    # RESUMEN FINAL
    # ==========================================
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL - ESTUDIANTE A")
    print("="*80)
    
    print("\n🔹 RED 1: PREDICCIÓN DE POTENCIAL MÁXIMO")
    print(f"   Mejores hiperparámetros:")
    print(f"     - Learning rate: {results_red1['best_params']['learning_rate']}")
    print(f"     - L2 lambda:     {results_red1['best_params']['l2_lambda']}")
    print(f"     - Iteraciones:   {results_red1['best_params']['n_iter']}")
    print(f"   Test RMSE: {results_red1['results']['test']['RMSE']:.3f}")
    print(f"   Test R²:   {results_red1['results']['test']['R2']:.4f}")
    
    print("\n🔹 RED 2: CLASIFICACIÓN DE PERFIL")
    print(f"   Mejores hiperparámetros:")
    print(f"     - Learning rate: {results_red2['best_params']['learning_rate']}")
    print(f"     - L2 lambda:     {results_red2['best_params']['l2_lambda']}")
    print(f"     - Iteraciones:   {results_red2['best_params']['n_iter']}")
    print(f"   Test Accuracy: {results_red2['results']['test_accuracy']:.3f}")
    
    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO Y OPTIMIZACIÓN COMPLETADOS")
    print("="*80)


if __name__ == "__main__":
    main()
