from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from collections import defaultdict


ID_COLS = {
    "id", "player_api_id", "player_fifa_api_id"
}

# =========================
# Utilidades generales
# =========================

def rmse(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def mae(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return float(np.mean(np.abs(y_pred - y_true)))

def r2_score(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

def train_test_kfold_indices(m, k=5, seed=42, shuffle=True):
    idx = np.arange(m)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train_idx, test_idx

def confusion_matrix(y_true, y_pred, labels):
    lab2i = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[lab2i[yt], lab2i[yp]] += 1
    return cm

# =========================
# VIF (sin statsmodels)
# VIF_j = 1 / (1 - R^2_j)
# donde R^2_j se calcula regresando X_j ~ X_otros
# =========================
def compute_vif_numpy(X, feature_names):
    # X: (m, n) sin bias, asume float
    m, n = X.shape
    vifs = []
    for j in range(n):
        y = X[:, j]
        Xo = np.delete(X, j, axis=1)

        # agregar bias para OLS de y ~ Xo
        Xo_b = np.c_[np.ones((m, 1)), Xo]
        # theta = pinv(X'X) X'y
        theta = np.linalg.pinv(Xo_b.T @ Xo_b) @ (Xo_b.T @ y.reshape(-1, 1))
        yhat = (Xo_b @ theta).reshape(-1)

        r2 = r2_score(y, yhat)
        vif = np.inf if (1 - r2) <= 1e-12 else 1.0 / (1.0 - r2)
        vifs.append((feature_names[j], float(vif), float(r2)))
    vifs.sort(key=lambda t: t[1], reverse=True)
    return vifs

def drop_high_vif(X, feature_names, threshold=10.0, min_features=10):
    Xc = X.copy()
    names = feature_names[:]
    while Xc.shape[1] > min_features:
        vifs = compute_vif_numpy(Xc, names)
        worst_name, worst_vif, _ = vifs[0]
        if worst_vif <= threshold or (not np.isinf(worst_vif) and worst_vif <= threshold):
            break
        # drop worst
        j = names.index(worst_name)
        Xc = np.delete(Xc, j, axis=1)
        names.pop(j)
    return Xc, names

# =========================
# Regresión Lineal desde cero
# =========================
class RegresionLineal:
    def __init__(self):
        self.Xb = None
        self.y = None
        self.theta_ = None
        self.mu_ = None
        self.std_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        m = X.shape[0]
        self.Xb = np.c_[np.ones((m, 1)), X]
        self.y = y
        self.theta_ = np.zeros((self.Xb.shape[1], 1))

    def normalizar(self):
        # z-score solo en features (no bias)
        self.mu_ = self.Xb[:, 1:].mean(axis=0)
        self.std_ = self.Xb[:, 1:].std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        self.Xb[:, 1:] = (self.Xb[:, 1:] - self.mu_) / self.std_

    def ecuacion_normal(self, l2_lambda=0.0):
        # Ridge opcional con lambda (si l2_lambda>0)
        X = self.Xb
        y = self.y
        n = X.shape[1]
        I = np.eye(n)
        I[0, 0] = 0  # no regularizar bias
        self.theta_ = np.linalg.pinv(X.T @ X + l2_lambda * I) @ (X.T @ y)

    def predecir(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.mu_ is not None and self.std_ is not None:
            X = (X - self.mu_) / self.std_

        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        return (Xb @ self.theta_).reshape(-1)

# =========================
# Regresión Logística binaria (para OVR) desde cero
# =========================
class LogRegBinaria:
    def __init__(self):
        self.w = None
        self.mu_ = None
        self.std_ = None

    def _sigmoid(self, z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y, lr=0.1, iters=5000, l2=1e-3, eps=1e-9, normalize=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        m, n = X.shape

        if normalize:
            self.mu_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
            X = (X - self.mu_) / self.std_

        Xb = np.c_[np.ones((m, 1)), X]
        self.w = np.zeros((n + 1, 1), dtype=float)

        prev = None
        for _ in range(iters):
            p = self._sigmoid(Xb @ self.w)  # (m,1)

            err = p - y
            grad = (Xb.T @ err) / m

            # L2 (no regularizar bias)
            reg = (l2 / m) * self.w
            reg[0, 0] = 0.0
            grad = grad + reg

            self.w = self.w - lr * grad

            # early stop por pérdida
            p_clip = np.clip(p, 1e-12, 1 - 1e-12)
            loss = -float(np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)))
            loss += float(l2 / (2 * m) * np.sum(self.w[1:] ** 2))
            if prev is not None and abs(prev - loss) < eps:
                break
            prev = loss

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.mu_ is not None and self.std_ is not None:
            X = (X - self.mu_) / self.std_

        Xb = np.c_[np.ones((X.shape[0], 1)), X]
        return self._sigmoid(Xb @ self.w).reshape(-1)

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X) >= thr).astype(int)

# =========================
# One-vs-Rest Multiclase
# =========================
class OVRLogistic:
    def __init__(self):
        self.models = {}
        self.classes_ = None

    def fit(self, X, y, **kwargs):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.models = {}
        for c in self.classes_:
            y_bin = (y == c).astype(int)
            m = LogRegBinaria()
            m.fit(X, y_bin, **kwargs)
            self.models[c] = m

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        scores = []
        for c in self.classes_:
            scores.append(self.models[c].predict_proba(X))
        P = np.vstack(scores).T  # (m, K)

        # normalizar para que sumen 1
        s = P.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return P / s

    def predict(self, X):
        P = self.predict_proba(X)
        idx = np.argmax(P, axis=1)
        return self.classes_[idx]

# =========================
# (C) SOLUCIÓN C: target posición por reglas (usa columnas del dataset)
# =========================
def build_position_target_by_rules(df: pd.DataFrame):
    required = [
        "gk_diving","gk_handling","gk_kicking","gk_positioning","gk_reflexes",
        "marking","standing_tackle","sliding_tackle","interceptions",
        "short_passing","long_passing","vision","ball_control",
        "finishing","positioning","shot_power","volleys","dribbling"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias para reglas de posición: {missing}")

    def col(name):
        return df[name].astype(float).to_numpy()

    # Scores por rol (0..100 típico)
    GK = (col("gk_diving") + col("gk_handling") + col("gk_kicking") +
          col("gk_positioning") + col("gk_reflexes")) / 5.0

    DEF = (col("marking") + col("standing_tackle") + col("sliding_tackle") +
           col("interceptions")) / 4.0

    MID = (col("short_passing") + col("long_passing") + col("vision") +
           col("ball_control")) / 4.0

    ATT = (col("finishing") + col("positioning") + col("shot_power") +
           col("volleys") + col("dribbling")) / 5.0

    scores = np.vstack([GK, DEF, MID, ATT]).T  # (m,4)
    labels = np.array(["Portero", "Defensa", "Medio", "Atacante"], dtype=object)

    y = labels[np.argmax(scores, axis=1)]

    # “híbridos” por regla: top1-top2 pequeño (en escala 0..100)
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    hybrid = margin < 5.0  # ajustable
    return y, hybrid, scores, labels

def build_position_target_7_classes(df: pd.DataFrame):
    """
    Genera 7 clases específicas para MLP Clasificador.
    
    Posiciones:
    1. Portero (GK)
    2. Defensa Central (CB) 
    3. Lateral (LB/RB)
    4. Pivote (CDM)
    5. Mediocentro (CM)
    6. Extremo (LW/RW)
    7. Delantero (ST)
    """
    required = [
        "gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes",
        "marking", "standing_tackle", "sliding_tackle", "interceptions",
        "short_passing", "long_passing", "vision", "ball_control",
        "finishing", "positioning", "shot_power", "volleys", "dribbling",
        "crossing", "acceleration", "sprint_speed", "agility"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias para 7 posiciones: {missing}")
    
    def col(name):
        return df[name].astype(float).to_numpy()
    
    # 1. Portero
    GK = (col("gk_diving") + col("gk_handling") + col("gk_kicking") +
          col("gk_positioning") + col("gk_reflexes")) / 5.0
    
    # 2. Defensa Central
    CB = (col("marking") + col("standing_tackle") + col("sliding_tackle") +
          col("interceptions")) / 4.0
    
    # 3. Lateral (defensa + velocidad + cruce)
    LB = (col("marking") + col("standing_tackle") + col("acceleration") +
          col("sprint_speed") + col("crossing")) / 5.0
    
    # 4. Pivote (defensa + pase)
    CDM = (col("interceptions") + col("standing_tackle") + col("short_passing") +
           col("long_passing") + col("marking")) / 5.0
    
    # 5. Mediocentro (pase + control)
    CM = (col("short_passing") + col("long_passing") + col("vision") +
          col("ball_control")) / 4.0
    
    # 6. Extremo (velocidad + regate + cruce)
    LW = (col("dribbling") + col("crossing") + col("acceleration") +
          col("sprint_speed") + col("agility")) / 5.0
    
    # 7. Delantero
    ST = (col("finishing") + col("positioning") + col("shot_power") +
          col("volleys") + col("dribbling")) / 5.0
    
    scores = np.vstack([GK, CB, LB, CDM, CM, LW, ST]).T  # (m, 7)
    labels = np.array(["Portero", "Defensa_Central", "Lateral", "Pivote", 
                       "Mediocentro", "Extremo", "Delantero"], dtype=object)
    
    y = labels[np.argmax(scores, axis=1)]
    
    # Híbridos: margen pequeño entre top-2
    sorted_scores = np.sort(scores, axis=1)
    margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    hybrid = margin < 5.0
    
    return y, hybrid, scores, labels

def undersample_balance(X, y, seed=42):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_c = counts.min()
    idx_all = []
    for c in classes:
        idx_c = np.where(y == c)[0]
        pick = rng.choice(idx_c, size=min_c, replace=False)
        idx_all.append(pick)
    idx_all = np.concatenate(idx_all)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]

# =========================
# MAIN
# =========================
def main():
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
    TARGET = "overall_rating"

    # --- Leer
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE};", conn)
    conn.close()
    print("[INFO] shape:", df.shape)

    if TARGET not in df.columns:
        raise ValueError(f"No existe {TARGET} en la tabla limpia.")

    # --- Selección top-15 por correlación abs
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    candidates = [c for c in num_cols if c not in ID_COLS and c != TARGET]
    corrs = df[candidates].corrwith(df[TARGET]).abs().sort_values(ascending=False)
    top15 = corrs.head(15).index.tolist()
    print("\n[REG] Top 15 por |corr| con overall_rating:")
    for i, c in enumerate(top15, 1):
        print(f"  {i:02d}. {c} | |corr|={corrs[c]:.4f}")

    X = df[top15].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)

    # --- VIF y reducción
    X_vif, feats_vif = drop_high_vif(X, top15, threshold=10.0, min_features=10)
    print("\n[REG] Features finales post-VIF:", feats_vif)

    # =========================
    # (B) K-Fold Regresión Lineal (OLS) + Ridge
    # =========================
    k = 5
    metrics_ols = []
    metrics_ridge = []

    for fold, (tr, te) in enumerate(train_test_kfold_indices(len(y), k=k, seed=42), 1):
        Xtr, ytr = X_vif[tr], y[tr]
        Xte, yte = X_vif[te], y[te]

        # OLS
        ols = RegresionLineal()
        ols.fit(Xtr, ytr)
        ols.normalizar()
        ols.ecuacion_normal(l2_lambda=0.0)
        pred = ols.predecir(Xte)
        metrics_ols.append((rmse(yte, pred), mae(yte, pred), r2_score(yte, pred)))

        # Ridge
        ridge = RegresionLineal()
        ridge.fit(Xtr, ytr)
        ridge.normalizar()
        ridge.ecuacion_normal(l2_lambda=1.0)  # puedes probar 0.1, 1, 10
        pred_r = ridge.predecir(Xte)
        metrics_ridge.append((rmse(yte, pred_r), mae(yte, pred_r), r2_score(yte, pred_r)))

        print(f"\nFold {fold}/{k}")
        print(f"  OLS   -> RMSE={metrics_ols[-1][0]:.3f} MAE={metrics_ols[-1][1]:.3f} R2={metrics_ols[-1][2]:.4f}")
        print(f"  Ridge -> RMSE={metrics_ridge[-1][0]:.3f} MAE={metrics_ridge[-1][1]:.3f} R2={metrics_ridge[-1][2]:.4f}")

    def summarize(ms):
        a = np.array(ms, dtype=float)
        return a.mean(axis=0), a.std(axis=0)

    mean_ols, std_ols = summarize(metrics_ols)
    mean_r, std_r = summarize(metrics_ridge)

    print("\n=== (B) RESULTADOS CV (k=5) ===")
    print(f"OLS   -> RMSE={mean_ols[0]:.3f}±{std_ols[0]:.3f} | MAE={mean_ols[1]:.3f} | R2={mean_ols[2]:.4f}")
    print(f"Ridge -> RMSE={mean_r[0]:.3f}±{std_r[0]:.3f} | MAE={mean_r[1]:.3f} | R2={mean_r[2]:.4f}")

    # Entrenar modelo final
    use_ridge = mean_r[0] < mean_ols[0]
    final = RegresionLineal()
    final.fit(X_vif, y)
    final.normalizar()
    final.ecuacion_normal(l2_lambda=1.0 if use_ridge else 0.0)
    print(f"\n[REG] Modelo final elegido: {'Ridge' if use_ridge else 'OLS'}")

    # Coeficientes e importancia relativa
    theta = final.theta_.reshape(-1)
    coef = theta[1:]
    rel = np.abs(coef)
    rel = rel / (rel.sum() if rel.sum() != 0 else 1) * 100.0
    coef_df = pd.DataFrame({"feature": feats_vif, "coef": coef, "importance_%": rel}).sort_values("importance_%", ascending=False)
    print("\n[REG] Importancia relativa (|coef|):")
    print(coef_df.head(10).to_string(index=False))

    coef_df.to_csv("regression_coefficients_importance.csv", index=False)

    # =========================
    # (C) Regresión Logística Multiclase OVR (Solución C)
    # =========================
    y_cls, hybrid_rule, score_mat, role_labels = build_position_target_by_rules(df)

    # usamos mismas features que regresión (feats_vif) para consistencia
    X_cls = df[feats_vif].to_numpy(dtype=float)

    print("\n[LOG] Target por reglas creado (4 clases).")
    classes, counts = np.unique(y_cls, return_counts=True)
    print("[LOG] Distribución:", dict(zip(classes.tolist(), counts.tolist())))
    print(f"[LOG] Híbridos por regla (margen score top1-top2 < 5.0): {int(hybrid_rule.sum())}")

    # (opcional) balanceo para evitar que una clase domine
    balance = True
    if balance:
        X_cls, y_cls = undersample_balance(X_cls, y_cls, seed=42)
        classes, counts = np.unique(y_cls, return_counts=True)
        print("[LOG] Distribución tras balanceo:", dict(zip(classes.tolist(), counts.tolist())))

    all_true = []
    all_pred = []

    for fold, (tr, te) in enumerate(train_test_kfold_indices(len(y_cls), k=5, seed=42, shuffle=True), 1):
        Xtr, ytr = X_cls[tr], y_cls[tr]
        Xte, yte = X_cls[te], y_cls[te]

        ovr = OVRLogistic()
        ovr.fit(
            Xtr, ytr,
            lr=0.25, iters=7000,
            l2=5e-3, eps=1e-9,
            normalize=True
        )

        pred = ovr.predict(Xte)
        all_true.extend(yte.tolist())
        all_pred.extend(pred.tolist())

        print(f"[LOG] Fold {fold}/5 listo.")

    classes = np.unique(y_cls)
    cm = confusion_matrix(all_true, all_pred, labels=list(classes))
    print("\n=== (C) MATRIZ DE CONFUSIÓN (real x pred) ===")
    print("Labels:", list(classes))
    print(cm)

    # Métricas por clase (precision, recall, f1) manual
    report = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
        report[c] = (precision, recall, f1)

    print("\n=== (C) MÉTRICAS POR CLASE ===")
    for c in classes:
        p, r, f1 = report[c]
        print(f"{c:10s} | Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

    acc = np.mean(np.array(all_true) == np.array(all_pred))
    print(f"\nAccuracy global: {acc:.4f}")

    # Entrenamiento final en todo (para probabilidades / híbridos por probas)
    final_ovr = OVRLogistic()
    final_ovr.fit(X_cls, y_cls, lr=0.25, iters=9000, l2=5e-3, eps=1e-9, normalize=True)

    P = final_ovr.predict_proba(X_cls)
    sortedP = np.sort(P, axis=1)
    marginP = sortedP[:, -1] - sortedP[:, -2]
    hybrid_idx = np.where(marginP < 0.10)[0]
    print(f"\nJugadores 'híbridos' (margen probas top1-top2 < 0.10): {len(hybrid_idx)}")
    if len(hybrid_idx) > 0:
        print("Ejemplos (primeros 10):")
        for i in hybrid_idx[:10]:
            order = np.argsort(P[i])[::-1]
            c1, c2 = final_ovr.classes_[order[0]], final_ovr.classes_[order[1]]
            print(f"  #{i}: {c1}={P[i,order[0]]:.3f} vs {c2}={P[i,order[1]]:.3f} margen={marginP[i]:.3f}")

    # Guardar outputs simples
    pd.DataFrame(cm, index=[f"real_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]) \
      .to_csv("logreg_confusion_matrix.csv", index=True)

    metrics_rows = [{"class": c, "precision": report[c][0], "recall": report[c][1], "f1": report[c][2]} for c in classes]
    pd.DataFrame(metrics_rows).to_csv("logreg_metrics_by_class.csv", index=False)
    print("\n[OK] Archivos generados:")
    print(" - regression_coefficients_importance.csv")
    print(" - logreg_confusion_matrix.csv")
    print(" - logreg_metrics_by_class.csv")

if __name__ == "__main__":
    main()
# ====================================
# MLP 1: Predicción de Potencial Máximo (REGRESIÓN)
# Arquitectura: 20-256-128-64-1
# Activación: ReLU (ocultas) + Lineal (salida)
# Regularización: L2 (λ)
# ====================================
class MLPRegressor:
    """
    Red Neuronal Multicapa para REGRESIÓN.
    Predice valores continuos (potencial máximo).
    
    Arquitectura especificada: 20 → 256 → 128 → 64 → 1
    - Capas ocultas: ReLU
    - Capa salida: Lineal (sin activación)
    - Regularización: L2 para prevenir overfitting
    """
    
    def __init__(self, layer_sizes, learning_rate=0.001, n_iter=2000, 
                 l2_lambda=0.01, verbose=True, batch_size=None):
        """
        Parámetros:
        -----------
        layer_sizes : list
            Ejemplo: [20, 256, 128, 64, 1] para Red 1
        learning_rate : float
            Tasa de aprendizaje (0.001 recomendado para redes profundas)
        n_iter : int
            Épocas de entrenamiento
        l2_lambda : float
            Parámetro de regularización L2 (λ)
        verbose : bool
            Mostrar progreso
        batch_size : int o None
            Tamaño del mini-batch (None = batch completo)
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2_lambda = l2_lambda
        self.verbose = verbose
        self.batch_size = batch_size
        self.parameters = {}
        self.costs = []
        self.mu_ = None  # Para normalización
        self.std_ = None
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Inicialización He para ReLU (mejora sobre Xavier)"""
        np.random.seed(42)
        for l in range(1, len(self.layer_sizes)):
            # He initialization: scale = sqrt(2 / n_in)
            scale = np.sqrt(2.0 / self.layer_sizes[l-1])
            self.parameters[f'W{l}'] = np.random.randn(
                self.layer_sizes[l], self.layer_sizes[l-1]) * scale
            self.parameters[f'b{l}'] = np.zeros((self.layer_sizes[l], 1))
    
    def normalize_data(self, X, fit=True):
        """Normalización Z-score"""
        if fit:
            self.mu_ = np.mean(X, axis=0, keepdims=True)
            self.std_ = np.std(X, axis=0, keepdims=True)
            self.std_[self.std_ == 0] = 1.0
        return (X - self.mu_) / self.std_
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def forward_propagation(self, X):
        """Propagación hacia adelante (salida LINEAL)"""
        cache = {'A0': X}
        A = X
        L = len(self.layer_sizes) - 1
        
        # Capas ocultas con ReLU
        for l in range(1, L):
            Z = self.parameters[f'W{l}'].dot(A) + self.parameters[f'b{l}']
            A = self.relu(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        
        # Capa de salida LINEAL (sin activación)
        Z = self.parameters[f'W{L}'].dot(A) + self.parameters[f'b{L}']
        A = Z  # ¡ACTIVACIÓN LINEAL!
        cache[f'Z{L}'] = Z
        cache[f'A{L}'] = A
        
        return A, cache
    
    def compute_cost(self, AL, Y):
        """MSE + Regularización L2"""
        m = Y.shape[1]
        
        # Error cuadrático medio
        mse = np.mean((AL - Y) ** 2)
        
        # Regularización L2: λ/(2m) * Σ(W²)
        l2_cost = 0
        L = len(self.layer_sizes) - 1
        for l in range(1, L + 1):
            l2_cost += np.sum(self.parameters[f'W{l}'] ** 2)
        l2_cost = (self.l2_lambda / (2 * m)) * l2_cost
        
        return mse + l2_cost
    
    def backward_propagation(self, AL, Y, cache):
        """Backpropagation con regularización L2"""
        m = Y.shape[1]
        L = len(self.layer_sizes) - 1
        grads = {}
        
        # Gradiente capa de salida (regresión: dL/dA = 2(A-Y))
        dA = 2 * (AL - Y) / m
        dZ = dA  # Derivada de activación lineal = 1
        
        grads[f'dW{L}'] = dZ.dot(cache[f'A{L-1}'].T) + (self.l2_lambda / m) * self.parameters[f'W{L}']
        grads[f'db{L}'] = np.sum(dZ, axis=1, keepdims=True)
        grads[f'dZ{L}'] = dZ
        
        # Capas ocultas
        for l in reversed(range(1, L)):
            dA = self.parameters[f'W{l+1}'].T.dot(grads[f'dZ{l+1}'])
            dZ = dA * self.relu_derivative(cache[f'Z{l}'])
            
            grads[f'dW{l}'] = dZ.dot(cache[f'A{l-1}'].T) + (self.l2_lambda / m) * self.parameters[f'W{l}']
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True)
            grads[f'dZ{l}'] = dZ
        
        return grads
    
    def update_parameters(self, grads):
        """Gradient descent"""
        L = len(self.layer_sizes) - 1
        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']
    
    def fit(self, X, y):
        """Entrenar red de regresión"""
        if self.verbose:
            print("\n" + "="*70)
            print("RED 1: PREDICCIÓN DE POTENCIAL MÁXIMO (REGRESIÓN)")
            print(f"Arquitectura: {'-'.join(map(str, self.layer_sizes))}")
            print(f"Regularización L2: λ = {self.l2_lambda}")
            print("="*70)
        
        # Normalizar datos
        X_norm = self.normalize_data(X, fit=True)
        X_norm = X_norm.T
        Y = y.reshape(1, -1)
        
        self.costs = []
        m = X_norm.shape[1]
        
        for i in range(self.n_iter):
            # Forward
            AL, cache = self.forward_propagation(X_norm)
            
            # Costo
            cost = self.compute_cost(AL, Y)
            self.costs.append(cost)
            
            # Backward
            grads = self.backward_propagation(AL, Y, cache)
            
            # Update
            self.update_parameters(grads)
            
            if self.verbose and i % 200 == 0:
                # Calcular RMSE para interpretabilidad
                rmse_val = np.sqrt(np.mean((AL - Y) ** 2))
                print(f"  Época {i:4d}: Costo={cost:.4f} | RMSE={rmse_val:.3f}")
        
        if self.verbose:
            final_rmse = np.sqrt(np.mean((AL - Y) ** 2))
            print(f"\n✓ Entrenamiento completado | RMSE final={final_rmse:.3f}")
    
    def predict(self, X):
        """Predecir valores continuos"""
        X_norm = self.normalize_data(X, fit=False)
        X_norm = X_norm.T
        AL, _ = self.forward_propagation(X_norm)
        return AL.reshape(-1)
    
    def score(self, X, y):
        """Calcular R² y RMSE"""
        predictions = self.predict(X)
        r2 = r2_score(y, predictions)
        rmse_val = rmse(y, predictions)
        mae_val = mae(y, predictions)
        return {'R2': r2, 'RMSE': rmse_val, 'MAE': mae_val}


# ====================================
# MLP 2: Clasificación de Perfil de Jugador
# Arquitectura: 15-256-128-7
# Activación: ReLU (ocultas) + Softmax (salida)
# Regularización: L2 (λ)
# ====================================
class MLPClassifier:
    """
    Red Neuronal Multicapa para CLASIFICACIÓN MULTICLASE.
    Predice posición del jugador (7 clases).
    
    Arquitectura especificada: 15 → 256 → 128 → 7
    - Capas ocultas: ReLU
    - Capa salida: Softmax
    - Regularización: L2
    """
    
    def __init__(self, layer_sizes, learning_rate=0.001, n_iter=2000, 
                 l2_lambda=0.01, verbose=True):
        """
        Parámetros:
        -----------
        layer_sizes : list
            Ejemplo: [15, 256, 128, 7] para Red 2
        learning_rate : float
            Tasa de aprendizaje
        n_iter : int
            Épocas de entrenamiento
        l2_lambda : float
            Regularización L2
        verbose : bool
            Mostrar progreso
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.l2_lambda = l2_lambda
        self.verbose = verbose
        self.parameters = {}
        self.costs = []
        self.classes_ = None
        self.mu_ = None
        self.std_ = None
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Inicialización He para ReLU"""
        np.random.seed(42)
        for l in range(1, len(self.layer_sizes)):
            scale = np.sqrt(2.0 / self.layer_sizes[l-1])
            self.parameters[f'W{l}'] = np.random.randn(
                self.layer_sizes[l], self.layer_sizes[l-1]) * scale
            self.parameters[f'b{l}'] = np.zeros((self.layer_sizes[l], 1))
    
    def normalize_data(self, X, fit=True):
        """Normalización Z-score"""
        if fit:
            self.mu_ = np.mean(X, axis=0, keepdims=True)
            self.std_ = np.std(X, axis=0, keepdims=True)
            self.std_[self.std_ == 0] = 1.0
        return (X - self.mu_) / self.std_
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def softmax(self, Z):
        """Softmax estable numéricamente"""
        Z = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    def forward_propagation(self, X):
        """Propagación hacia adelante (salida SOFTMAX)"""
        cache = {'A0': X}
        A = X
        L = len(self.layer_sizes) - 1
        
        # Capas ocultas con ReLU
        for l in range(1, L):
            Z = self.parameters[f'W{l}'].dot(A) + self.parameters[f'b{l}']
            A = self.relu(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        
        # Capa de salida con Softmax
        Z = self.parameters[f'W{L}'].dot(A) + self.parameters[f'b{L}']
        A = self.softmax(Z)
        cache[f'Z{L}'] = Z
        cache[f'A{L}'] = A
        
        return A, cache
    
    def compute_cost(self, AL, Y):
        """Cross-entropy + Regularización L2"""
        m = Y.shape[1]
        
        # Entropía cruzada
        cross_entropy = -np.sum(Y * np.log(AL + 1e-8)) / m
        
        # Regularización L2
        l2_cost = 0
        L = len(self.layer_sizes) - 1
        for l in range(1, L + 1):
            l2_cost += np.sum(self.parameters[f'W{l}'] ** 2)
        l2_cost = (self.l2_lambda / (2 * m)) * l2_cost
        
        return cross_entropy + l2_cost
    
    def backward_propagation(self, AL, Y, cache):
        """Backpropagation con regularización L2"""
        m = Y.shape[1]
        L = len(self.layer_sizes) - 1
        grads = {}
        
        # Gradiente capa de salida (softmax + cross-entropy)
        dZ = AL - Y
        grads[f'dW{L}'] = dZ.dot(cache[f'A{L-1}'].T) / m + (self.l2_lambda / m) * self.parameters[f'W{L}']
        grads[f'db{L}'] = np.sum(dZ, axis=1, keepdims=True) / m
        grads[f'dZ{L}'] = dZ
        
        # Capas ocultas
        for l in reversed(range(1, L)):
            dA = self.parameters[f'W{l+1}'].T.dot(grads[f'dZ{l+1}'])
            dZ = dA * self.relu_derivative(cache[f'Z{l}'])
            
            grads[f'dW{l}'] = dZ.dot(cache[f'A{l-1}'].T) / m + (self.l2_lambda / m) * self.parameters[f'W{l}']
            grads[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True) / m
            grads[f'dZ{l}'] = dZ
        
        return grads
    
    def update_parameters(self, grads):
        """Gradient descent"""
        L = len(self.layer_sizes) - 1
        for l in range(1, L + 1):
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}']
    
    def fit(self, X, y):
        """Entrenar red de clasificación"""
        if self.verbose:
            print("\n" + "="*70)
            print("RED 2: CLASIFICACIÓN DE PERFIL DE JUGADOR")
            print(f"Arquitectura: {'-'.join(map(str, self.layer_sizes))}")
            print(f"Regularización L2: λ = {self.l2_lambda}")
            print("="*70)
        
        # Normalizar datos
        X_norm = self.normalize_data(X, fit=True)
        X_norm = X_norm.T
        
        # Preparar clases y one-hot encoding
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes != self.layer_sizes[-1]:
            raise ValueError(f"El número de clases ({n_classes}) no coincide con neuronas de salida ({self.layer_sizes[-1]})")
        
        # One-hot encoding
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y])
        Y = np.zeros((n_classes, len(y)))
        Y[y_idx, np.arange(len(y))] = 1
        
        self.costs = []
        
        for i in range(self.n_iter):
            # Forward
            AL, cache = self.forward_propagation(X_norm)
            
            # Costo
            cost = self.compute_cost(AL, Y)
            self.costs.append(cost)
            
            # Backward
            grads = self.backward_propagation(AL, Y, cache)
            
            # Update
            self.update_parameters(grads)
            
            if self.verbose and i % 200 == 0:
                # Calcular accuracy
                pred = np.argmax(AL, axis=0)
                acc = np.mean(pred == y_idx)
                print(f"  Época {i:4d}: Costo={cost:.4f} | Accuracy={acc:.3f}")
        
        if self.verbose:
            pred = np.argmax(AL, axis=0)
            final_acc = np.mean(pred == y_idx)
            print(f"\n✓ Entrenamiento completado | Accuracy final={final_acc:.3f}")
    
    def predict_proba(self, X):
        """Predecir probabilidades"""
        X_norm = self.normalize_data(X, fit=False)
        X_norm = X_norm.T
        AL, _ = self.forward_propagation(X_norm)
        return AL.T
    
    def predict(self, X):
        """Predecir clases"""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]
    
    def accuracy(self, X, y):
        """Calcular exactitud"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def confusion_matrix(self, X, y):
        """Generar matriz de confusión"""
        predictions = self.predict(X)
        return confusion_matrix(y, predictions, self.classes_)