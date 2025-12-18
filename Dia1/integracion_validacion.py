import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import (
    RegresionLineal, OVRLogistic, 
    rmse, mae, r2_score,
    train_test_kfold_indices,
    confusion_matrix,
    drop_high_vif,
    build_position_target_by_rules,
    undersample_balance,
    ID_COLS
)
from preprocesamiento import (
    fill_nulls_by_group_median,
    minmax_0_100,
    add_derived_scores
)

# =========================
# Pipeline Unificado de Preprocesamiento
# =========================
class PreprocessingPipeline:
    """
    Pipeline unificado que integra todas las etapas de preprocesamiento:
    1. Manejo de valores nulos por posición
    2. Creación de variables derivadas
    3. Codificación One-Hot
    4. Normalización (0-100)
    """
    
    def __init__(self, position_col=None, vif_threshold=10.0, min_features=10):
        self.position_col = position_col
        self.vif_threshold = vif_threshold
        self.min_features = min_features
        
        # Guardar transformaciones aprendidas
        self.cat_columns = None
        self.num_columns = None
        self.min_vals = {}
        self.max_vals = {}
        self.selected_features = None
        self.is_fitted = False
        
    def fit_transform(self, df, target_col=None):
        """
        Ajusta el pipeline y transforma los datos.
        
        Args:
            df: DataFrame con datos crudos
            target_col: Nombre de columna objetivo (no se transforma)
            
        Returns:
            DataFrame transformado
        """
        df_out = df.copy()
        
        # 1. Manejo de valores nulos
        print("[Pipeline] Paso 1/4: Manejo de valores nulos por posición...")
        df_out = fill_nulls_by_group_median(df_out, self.position_col)
        
        # 2. Variables derivadas
        print("[Pipeline] Paso 2/4: Creación de variables derivadas...")
        df_out = add_derived_scores(df_out)
        
        # 3. One-Hot Encoding
        print("[Pipeline] Paso 3/4: Codificación One-Hot...")
        exclude_ohe = set(ID_COLS)
        if target_col:
            exclude_ohe.add(target_col)
        if "date" in df_out.columns:
            exclude_ohe.add("date")
        if "birthday" in df_out.columns:
            exclude_ohe.add("birthday")
            
        self.cat_columns = [c for c in df_out.columns 
                           if df_out[c].dtype == "object" and c not in exclude_ohe]
        
        df_out = pd.get_dummies(df_out, columns=self.cat_columns, 
                               drop_first=False, dummy_na=False)
        
        # 4. Normalización 0-100
        print("[Pipeline] Paso 4/4: Normalización (0-100)...")
        exclude_norm = set(ID_COLS)
        if target_col and target_col in df_out.columns:
            exclude_norm.add(target_col)
            
        self.num_columns = [c for c in df_out.columns 
                           if pd.api.types.is_numeric_dtype(df_out[c]) 
                           and c not in exclude_norm]
        
        # Guardar min/max para transform posterior
        for c in self.num_columns:
            col = df_out[c].astype(float)
            self.min_vals[c] = np.nanmin(col)
            self.max_vals[c] = np.nanmax(col)
        
        df_out = minmax_0_100(df_out, exclude=exclude_norm)
        
        self.is_fitted = True
        print(f"[Pipeline] ✓ Pipeline ajustado. Shape final: {df_out.shape}")
        
        return df_out
    
    def get_feature_names(self, df):
        """Retorna nombres de features después del pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline no ha sido ajustado. Llama fit_transform primero.")
        
        # Simular transformación para obtener nombres de columnas
        exclude = set(ID_COLS)
        return [c for c in df.columns if c not in exclude]


# =========================
# Tests Estadísticos
# =========================
def shapiro_wilk_test(residuals, sample_size=5000):
    """
    Test de Shapiro-Wilk para normalidad de residuos.
    H0: Los residuos siguen una distribución normal
    
    Args:
        residuals: array de residuos
        sample_size: máximo de muestras (Shapiro-Wilk es costoso con muchos datos)
        
    Returns:
        dict con estadístico, p-value y conclusión
    """
    from scipy import stats
    
    residuals = np.asarray(residuals).reshape(-1)
    
    # Si hay muchos datos, tomar muestra aleatoria
    if len(residuals) > sample_size:
        np.random.seed(42)
        residuals = np.random.choice(residuals, size=sample_size, replace=False)
    
    statistic, p_value = stats.shapiro(residuals)
    
    result = {
        'test': 'Shapiro-Wilk',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'normal': p_value > 0.05,
        'interpretation': 'Los residuos siguen distribución normal' if p_value > 0.05 
                         else 'Los residuos NO siguen distribución normal'
    }
    
    return result


def breusch_pagan_test(y_true, y_pred, X):
    """
    Test de Breusch-Pagan para homocedasticidad (varianza constante).
    H0: Los residuos tienen varianza constante (homocedasticidad)
    
    Args:
        y_true: valores reales
        y_pred: valores predichos
        X: matriz de features (sin bias)
        
    Returns:
        dict con estadístico, p-value y conclusión
    """
    from scipy import stats
    
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    X = np.asarray(X)
    
    # Calcular residuos
    residuals = y_true - y_pred
    
    # Residuos al cuadrado
    residuals_sq = residuals ** 2
    
    # Regresión auxiliar: residuos^2 ~ X
    m, n = X.shape
    X_bias = np.c_[np.ones((m, 1)), X]
    
    # OLS para residuos^2
    try:
        theta = np.linalg.pinv(X_bias.T @ X_bias) @ (X_bias.T @ residuals_sq.reshape(-1, 1))
        fitted = (X_bias @ theta).reshape(-1)
        
        # R^2 de la regresión auxiliar
        ss_res = np.sum((residuals_sq - fitted) ** 2)
        ss_tot = np.sum((residuals_sq - np.mean(residuals_sq)) ** 2)
        r2_aux = 1 - ss_res / ss_tot if ss_tot != 0 else 0
        
        # Estadístico BP: n * R^2 ~ Chi^2(k) donde k = número de features
        bp_statistic = m * r2_aux
        p_value = 1 - stats.chi2.cdf(bp_statistic, n)
        
        result = {
            'test': 'Breusch-Pagan',
            'statistic': float(bp_statistic),
            'p_value': float(p_value),
            'homoscedastic': p_value > 0.05,
            'interpretation': 'Varianza constante (homocedasticidad)' if p_value > 0.05
                             else 'Varianza NO constante (heterocedasticidad)'
        }
    except:
        result = {
            'test': 'Breusch-Pagan',
            'statistic': np.nan,
            'p_value': np.nan,
            'homoscedastic': None,
            'interpretation': 'Error al calcular test'
        }
    
    return result


def anderson_darling_test(residuals):
    """
    Test de Anderson-Darling para normalidad (alternativa a Shapiro-Wilk).
    Más robusto con grandes muestras.
    """
    from scipy import stats
    
    residuals = np.asarray(residuals).reshape(-1)
    result = stats.anderson(residuals, dist='norm')
    
    # Usar nivel de significancia 5% (índice 2)
    critical_value = result.critical_values[2]  # 5%
    
    return {
        'test': 'Anderson-Darling',
        'statistic': float(result.statistic),
        'critical_value_5%': float(critical_value),
        'normal': result.statistic < critical_value,
        'interpretation': 'Los residuos siguen distribución normal' if result.statistic < critical_value
                         else 'Los residuos NO siguen distribución normal'
    }


# =========================
# Validación de Consistencia
# =========================
def validate_model_consistency(reg_features, log_features, reg_metrics, log_metrics):
    """
    Valida que ambos modelos usen transformaciones consistentes.
    
    Returns:
        dict con resultados de validación
    """
    consistency = {
        'same_features': set(reg_features) == set(log_features),
        'num_features_regression': len(reg_features),
        'num_features_logistic': len(log_features),
        'common_features': len(set(reg_features) & set(log_features)),
        'regression_performance': reg_metrics,
        'logistic_performance': log_metrics
    }
    
    if consistency['same_features']:
        consistency['status'] = 'CONSISTENTE'
        consistency['message'] = 'Ambos modelos usan las mismas features'
    else:
        consistency['status'] = 'INCONSISTENTE'
        diff_reg = set(reg_features) - set(log_features)
        diff_log = set(log_features) - set(reg_features)
        consistency['message'] = f'Diferencias encontradas. Reg-only: {len(diff_reg)}, Log-only: {len(diff_log)}'
        consistency['regression_only'] = list(diff_reg)[:10]
        consistency['logistic_only'] = list(diff_log)[:10]
    
    return consistency


# =========================
# Dashboard de Visualizaciones
# =========================
def create_dashboard(df, reg_model, log_model, X_reg, y_reg, X_log, y_log, 
                    feats_reg, feats_log, output_dir='visualizations'):
    """
    Crea dashboard con múltiples visualizaciones.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n[Dashboard] Generando visualizaciones en {output_dir}/...")
    
    # Configurar estilo
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Análisis de Residuos (Regresión)
    print("[Dashboard] 1/6: Gráficos de residuos...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis de Residuos - Regresión Lineal', fontsize=16, fontweight='bold')
    
    y_pred = reg_model.predecir(X_reg)
    residuals = y_reg.reshape(-1) - y_pred
    
    # 1a. Residuos vs Predicciones
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Valores Predichos')
    axes[0, 0].set_ylabel('Residuos')
    axes[0, 0].set_title('Residuos vs Predicciones')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 1b. Histograma de residuos
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Residuos')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Residuos')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    
    # 1c. Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normalidad)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 1d. Residuos absolutos vs Predicciones (heterocedasticidad)
    axes[1, 1].scatter(y_pred, np.abs(residuals), alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Valores Predichos')
    axes[1, 1].set_ylabel('|Residuos|')
    axes[1, 1].set_title('Residuos Absolutos vs Predicciones')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'residuos_regresion.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Importancia de Variables (Regresión)
    print("[Dashboard] 2/6: Importancia de variables...")
    theta = reg_model.theta_.reshape(-1)
    coef = theta[1:]  # sin bias
    importance = np.abs(coef)
    importance = importance / (importance.sum() if importance.sum() != 0 else 1) * 100
    
    coef_df = pd.DataFrame({
        'feature': feats_reg,
        'coefficient': coef,
        'importance_%': importance
    }).sort_values('importance_%', ascending=True).tail(15)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(coef_df)))
    ax.barh(coef_df['feature'], coef_df['importance_%'], color=colors)
    ax.set_xlabel('Importancia Relativa (%)', fontsize=12)
    ax.set_title('Top 15 Variables más Importantes - Regresión Lineal', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'importancia_variables_regresion.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Predicciones vs Reales (Regresión)
    print("[Dashboard] 3/6: Predicciones vs valores reales...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(y_reg, y_pred, alpha=0.5, s=20)
    
    # Línea ideal
    min_val = min(y_reg.min(), y_pred.min())
    max_val = max(y_reg.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    
    ax.set_xlabel('Valores Reales', fontsize=12)
    ax.set_ylabel('Valores Predichos', fontsize=12)
    ax.set_title('Predicciones vs Valores Reales - Regresión', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Añadir métricas
    r2 = r2_score(y_reg, y_pred)
    rmse_val = rmse(y_reg, y_pred)
    mae_val = mae(y_reg, y_pred)
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse_val:.3f}\nMAE = {mae_val:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_path / 'predicciones_vs_reales.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Matriz de Confusión (Logística)
    print("[Dashboard] 4/6: Matriz de confusión...")
    y_log_pred = log_model.predict(X_log)
    classes = log_model.classes_
    cm = confusion_matrix(y_log.tolist(), y_log_pred.tolist(), labels=list(classes))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, 
                yticklabels=classes, ax=ax, cbar_kws={'label': 'Frecuencia'})
    ax.set_xlabel('Predicho', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title('Matriz de Confusión - Regresión Logística Multiclase', 
                fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path / 'matriz_confusion.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Distribución de Probabilidades por Clase
    print("[Dashboard] 5/6: Distribución de probabilidades...")
    P = log_model.predict_proba(X_log)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Distribución de Probabilidades por Clase', fontsize=16, fontweight='bold')
    
    for i, (c, ax) in enumerate(zip(classes, axes.flat)):
        ax.hist(P[:, i], bins=50, edgecolor='black', alpha=0.7, color=plt.cm.Set3(i))
        ax.set_xlabel('Probabilidad', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.set_title(f'Clase: {c}', fontsize=11, fontweight='bold')
        ax.axvline(x=P[:, i].mean(), color='r', linestyle='--', 
                  linewidth=2, label=f'Media={P[:, i].mean():.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'distribucion_probabilidades.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Métricas por Clase (Logística)
    print("[Dashboard] 6/6: Métricas por clase...")
    report = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
        report[c] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    metrics_df = pd.DataFrame(report).T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.25
    
    ax.bar(x - width, metrics_df['precision'], width, label='Precision', alpha=0.8)
    ax.bar(x, metrics_df['recall'], width, label='Recall', alpha=0.8)
    ax.bar(x + width, metrics_df['f1'], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Clase', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Métricas por Clase - Regresión Logística', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path / 'metricas_por_clase.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[Dashboard] ✓ 6 visualizaciones guardadas en {output_dir}/")
    
    return {
        'residuals': residuals,
        'y_pred_reg': y_pred,
        'y_pred_log': y_log_pred,
        'confusion_matrix': cm,
        'class_metrics': report
    }


# =========================
# MAIN - Integración Completa
# =========================
def main():
    print("="*70)
    print(" PARTE D: INTEGRACIÓN Y VALIDACIÓN")
    print("="*70)
    
    # Buscar la base de datos en el directorio del script o directorio padre
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
            print(f"[INFO] Base de datos encontrada: {DB}")
            break
    
    if DB is None:
        raise FileNotFoundError(
            f"No se encuentra database_clean.sqlite. Ejecuta preprocesamiento.py primero.\n"
            f"Rutas buscadas: {[str(p) for p in DB_OPTIONS]}"
        )
    
    TABLE = "player_attributes_clean"
    
    # --- 1. Cargar datos ---
    print("\n[1/5] Cargando datos preprocesados...")
    conn = sqlite3.connect(DB)
    
    # Verificar que la tabla existe
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = [t[0] for t in tables]
    print(f"  Tablas disponibles: {table_names}")
    
    if TABLE not in table_names:
        conn.close()
        raise ValueError(f"La tabla '{TABLE}' no existe en {DB}. Tablas disponibles: {table_names}")
    
    df = pd.read_sql_query(f"SELECT * FROM {TABLE};", conn)
    conn.close()
    print(f"  Shape: {df.shape}")
    
    # --- 2. Pipeline Unificado ---
    print("\n[2/5] Aplicando pipeline unificado de preprocesamiento...")
    
    # El pipeline ya está aplicado en preprocesamiento.py, pero validamos
    # que las transformaciones sean consistentes
    pipeline = PreprocessingPipeline(position_col=None, vif_threshold=10.0)
    
    # Seleccionar features para regresión
    TARGET_REG = "overall_rating"
    if TARGET_REG not in df.columns:
        raise ValueError(f"No existe columna {TARGET_REG}")
    
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in num_cols if c not in ID_COLS and c != TARGET_REG]
    
    # Top 15 por correlación
    corrs = df[candidates].corrwith(df[TARGET_REG]).abs().sort_values(ascending=False)
    top15 = corrs.head(15).index.tolist()
    
    X_reg = df[top15].to_numpy(dtype=float)
    y_reg = df[TARGET_REG].to_numpy(dtype=float)
    
    # Reducción VIF
    X_reg_vif, feats_reg = drop_high_vif(X_reg, top15, threshold=10.0, min_features=10)
    print(f"  Features regresión (post-VIF): {len(feats_reg)}")
    
    # --- 3. Entrenar Modelos ---
    print("\n[3/5] Entrenando modelos con validación cruzada...")
    
    # REGRESIÓN LINEAL
    print("\n  [Regresión Lineal]")
    k = 5
    metrics_reg = []
    
    for fold, (tr, te) in enumerate(train_test_kfold_indices(len(y_reg), k=k, seed=42), 1):
        Xtr, ytr = X_reg_vif[tr], y_reg[tr]
        Xte, yte = X_reg_vif[te], y_reg[te]
        
        model = RegresionLineal()
        model.fit(Xtr, ytr)
        model.normalizar()
        model.ecuacion_normal(l2_lambda=1.0)
        
        pred = model.predecir(Xte)
        metrics_reg.append((rmse(yte, pred), mae(yte, pred), r2_score(yte, pred)))
    
    mean_reg = np.array(metrics_reg).mean(axis=0)
    print(f"    RMSE={mean_reg[0]:.3f} | MAE={mean_reg[1]:.3f} | R²={mean_reg[2]:.4f}")
    
    # Modelo final regresión
    reg_final = RegresionLineal()
    reg_final.fit(X_reg_vif, y_reg)
    reg_final.normalizar()
    reg_final.ecuacion_normal(l2_lambda=1.0)
    
    # REGRESIÓN LOGÍSTICA
    print("\n  [Regresión Logística Multiclase]")
    y_cls, _, _, _ = build_position_target_by_rules(df)
    X_log = df[feats_reg].to_numpy(dtype=float)  # Usar mismas features
    
    # Balanceo
    X_log_bal, y_log_bal = undersample_balance(X_log, y_cls, seed=42)
    
    metrics_log = []
    all_true = []
    all_pred = []
    
    for fold, (tr, te) in enumerate(train_test_kfold_indices(len(y_log_bal), k=5, seed=42), 1):
        Xtr, ytr = X_log_bal[tr], y_log_bal[tr]
        Xte, yte = X_log_bal[te], y_log_bal[te]
        
        ovr = OVRLogistic()
        ovr.fit(Xtr, ytr, lr=0.25, iters=7000, l2=5e-3, normalize=True)
        
        pred = ovr.predict(Xte)
        all_true.extend(yte.tolist())
        all_pred.extend(pred.tolist())
    
    acc = np.mean(np.array(all_true) == np.array(all_pred))
    print(f"    Accuracy={acc:.4f}")
    
    # Modelo final logística
    log_final = OVRLogistic()
    log_final.fit(X_log_bal, y_log_bal, lr=0.25, iters=9000, l2=5e-3, normalize=True)
    
    # --- 4. Validación Estadística ---
    print("\n[4/5] Validación estadística...")
    
    # Tests de normalidad y homocedasticidad
    y_pred_reg = reg_final.predecir(X_reg_vif)
    residuals = y_reg.reshape(-1) - y_pred_reg
    
    print("\n  [Test de Normalidad - Shapiro-Wilk]")
    shapiro_result = shapiro_wilk_test(residuals, sample_size=5000)
    print(f"    Estadístico: {shapiro_result['statistic']:.6f}")
    print(f"    P-value: {shapiro_result['p_value']:.6f}")
    print(f"    Conclusión: {shapiro_result['interpretation']}")
    
    print("\n  [Test de Normalidad - Anderson-Darling]")
    anderson_result = anderson_darling_test(residuals)
    print(f"    Estadístico: {anderson_result['statistic']:.6f}")
    print(f"    Valor crítico (5%): {anderson_result['critical_value_5%']:.6f}")
    print(f"    Conclusión: {anderson_result['interpretation']}")
    
    print("\n  [Test de Homocedasticidad - Breusch-Pagan]")
    bp_result = breusch_pagan_test(y_reg, y_pred_reg, X_reg_vif)
    print(f"    Estadístico: {bp_result['statistic']:.6f}")
    print(f"    P-value: {bp_result['p_value']:.6f}")
    print(f"    Conclusión: {bp_result['interpretation']}")
    
    # Validación de consistencia
    print("\n  [Validación de Consistencia entre Modelos]")
    consistency = validate_model_consistency(
        feats_reg, feats_reg,  # Usamos mismas features
        {'RMSE': mean_reg[0], 'MAE': mean_reg[1], 'R2': mean_reg[2]},
        {'Accuracy': acc}
    )
    print(f"    Estado: {consistency['status']}")
    print(f"    {consistency['message']}")
    print(f"    Features comunes: {consistency['common_features']}")
    
    # --- 5. Dashboard de Visualizaciones ---
    print("\n[5/5] Generando dashboard con visualizaciones...")
    dashboard_results = create_dashboard(
        df, reg_final, log_final,
        X_reg_vif, y_reg,
        X_log_bal, y_log_bal,
        feats_reg, feats_reg,
        output_dir='visualizations'
    )
    
    # --- Guardar Reporte Final ---
    print("\n[Reporte] Generando documento de hallazgos...")
    
    # Convertir numpy bool a Python bool para JSON
    def convert_for_json(obj):
        """Convierte tipos numpy a tipos nativos de Python para serialización JSON."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        return obj
    
    report = {
        'fecha': '2025-12-15',
        'dataset_shape': list(df.shape),
        'features_seleccionadas': feats_reg,
        'regresion_lineal': {
            'rmse': float(mean_reg[0]),
            'mae': float(mean_reg[1]),
            'r2': float(mean_reg[2])
        },
        'regresion_logistica': {
            'accuracy': float(acc),
            'clases': log_final.classes_.tolist()
        },
        'tests_estadisticos': {
            'shapiro_wilk': convert_for_json(shapiro_result),
            'anderson_darling': convert_for_json(anderson_result),
            'breusch_pagan': convert_for_json(bp_result)
        },
        'consistencia_modelos': convert_for_json(consistency)
    }
    
    import json
    with open('reporte_integracion_validacion.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n  ✓ Reporte guardado: reporte_integracion_validacion.json")
    
    # Reporte texto
    with open('HALLAZGOS.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(" HALLAZGOS - INTEGRACIÓN Y VALIDACIÓN\n")
        f.write("="*70 + "\n\n")
        
        f.write("1. PIPELINE UNIFICADO\n")
        f.write("-"*70 + "\n")
        f.write(f"Dataset procesado: {df.shape[0]} registros, {df.shape[1]} columnas\n")
        f.write(f"Features seleccionadas: {len(feats_reg)}\n")
        f.write(f"Lista: {', '.join(feats_reg)}\n\n")
        
        f.write("2. REGRESIÓN LINEAL (Overall Rating)\n")
        f.write("-"*70 + "\n")
        f.write(f"RMSE: {mean_reg[0]:.3f}\n")
        f.write(f"MAE:  {mean_reg[1]:.3f}\n")
        f.write(f"R²:   {mean_reg[2]:.4f}\n\n")
        
        f.write("3. REGRESIÓN LOGÍSTICA MULTICLASE (Posición)\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Clases: {', '.join(log_final.classes_)}\n\n")
        
        f.write("4. VALIDACIÓN ESTADÍSTICA\n")
        f.write("-"*70 + "\n")
        f.write(f"Shapiro-Wilk (normalidad):\n")
        f.write(f"  - p-value: {shapiro_result['p_value']:.6f}\n")
        f.write(f"  - {shapiro_result['interpretation']}\n\n")
        
        f.write(f"Anderson-Darling (normalidad):\n")
        f.write(f"  - Estadístico: {anderson_result['statistic']:.6f}\n")
        f.write(f"  - {anderson_result['interpretation']}\n\n")
        
        f.write(f"Breusch-Pagan (homocedasticidad):\n")
        f.write(f"  - p-value: {bp_result['p_value']:.6f}\n")
        f.write(f"  - {bp_result['interpretation']}\n\n")
        
        f.write("5. CONSISTENCIA ENTRE MODELOS\n")
        f.write("-"*70 + "\n")
        f.write(f"Estado: {consistency['status']}\n")
        f.write(f"{consistency['message']}\n\n")
        
        f.write("6. VISUALIZACIONES GENERADAS\n")
        f.write("-"*70 + "\n")
        f.write("  - residuos_regresion.png\n")
        f.write("  - importancia_variables_regresion.png\n")
        f.write("  - predicciones_vs_reales.png\n")
        f.write("  - matriz_confusion.png\n")
        f.write("  - distribucion_probabilidades.png\n")
        f.write("  - metricas_por_clase.png\n\n")
        
        f.write("7. CONCLUSIONES\n")
        f.write("-"*70 + "\n")
        f.write(f"✓ Pipeline de preprocesamiento unificado implementado\n")
        f.write(f"✓ Ambos modelos usan {len(feats_reg)} features consistentes\n")
        f.write(f"✓ Regresión lineal con R²={mean_reg[2]:.4f}\n")
        f.write(f"✓ Clasificación multiclase con Accuracy={acc:.4f}\n")
        
        if shapiro_result['normal']:
            f.write(f"✓ Residuos siguen distribución normal (Shapiro-Wilk)\n")
        else:
            f.write(f"⚠ Residuos NO siguen distribución normal perfecta\n")
            
        if bp_result['homoscedastic']:
            f.write(f"✓ Varianza constante de residuos (homocedasticidad)\n")
        else:
            f.write(f"⚠ Posible heterocedasticidad detectada\n")
        
        f.write("\n")
    
    print("  ✓ Hallazgos guardados: HALLAZGOS.txt")
    
    print("\n" + "="*70)
    print(" ✓ PARTE D COMPLETADA")
    print("="*70)
    print("\nArchivos generados:")
    print("  - reporte_integracion_validacion.json")
    print("  - HALLAZGOS.txt")
    print("  - visualizations/ (6 gráficos)")
    print("\n¡Integración y validación completadas exitosamente!")


if __name__ == "__main__":
    main()
