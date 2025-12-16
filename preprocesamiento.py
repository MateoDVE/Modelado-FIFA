import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
INPUT_DB = "database.sqlite"
OUTPUT_DB = "database_clean.sqlite"
OUTPUT_TABLE = "player_attributes_clean"

# Posibles nombres de columna "posición" (varía según dataset)
POSITION_CANDIDATES = [
    "position", "player_position", "preferred_position", "preferred_positions",
    "pos", "role", "club_position", "nation_position"
]

# IDs comunes (no normalizar ni one-hot)
ID_COLS = {
    "id", "player_api_id", "player_fifa_api_id"
}

# Scores derivados (si las columnas existen)
PHYSICAL_COLS = ["acceleration", "sprint_speed", "stamina", "strength"]
TECHNICAL_COLS = ["ball_control", "dribbling", "short_passing"]
MENTAL_COLS = ["positioning", "vision", "reactions"]

# -----------------------------
# Helpers
# -----------------------------
def list_tables(conn: sqlite3.Connection) -> list[str]:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    return [r[0] for r in conn.execute(q).fetchall()]

def pick_main_table(tables: list[str]) -> str:
    # FIFA sqlite clásico usa Player_Attributes como tabla de 42+ atributos
    if "Player_Attributes" in tables:
        return "Player_Attributes"
    # fallback razonable: la tabla más grande por filas (estimación)
    return tables[0]

def find_position_col(df: pd.DataFrame) -> str | None:
    for c in POSITION_CANDIDATES:
        if c in df.columns:
            return c
    return None

def safe_to_datetime(s: pd.Series) -> pd.Series:
    # maneja strings tipo '2015-02-20 00:00:00' o '2015-02-20'
    return pd.to_datetime(s, errors="coerce", utc=False)

def fill_nulls_by_group_median(df: pd.DataFrame, group_col: str | None) -> pd.DataFrame:
    out = df.copy()

    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c]) and c not in ID_COLS]
    cat_cols = [c for c in out.columns if out[c].dtype == "object" and c not in ID_COLS]

    if group_col is None:
        # numéricas: mediana global
        for c in num_cols:
            med = out[c].median()
            out[c] = out[c].fillna(med)
        # categóricas: moda global
        for c in cat_cols:
            if out[c].isna().any():
                mode = out[c].mode(dropna=True)
                out[c] = out[c].fillna(mode.iloc[0] if not mode.empty else "Unknown")
        return out

    # numéricas: mediana por grupo + fallback global
    for c in num_cols:
        global_med = out[c].median()
        out[c] = out[c].fillna(out.groupby(group_col)[c].transform("median"))
        out[c] = out[c].fillna(global_med)

    # categóricas: moda por grupo + fallback global
    for c in cat_cols:
        if out[c].isna().any():
            global_mode = out[c].mode(dropna=True)
            global_mode = global_mode.iloc[0] if not global_mode.empty else "Unknown"

            def group_mode(s: pd.Series):
                m = s.mode(dropna=True)
                return m.iloc[0] if not m.empty else global_mode

            out[c] = out[c].fillna(out.groupby(group_col)[c].transform(group_mode))
            out[c] = out[c].fillna(global_mode)

    return out

def minmax_0_100(df: pd.DataFrame, exclude: set[str]) -> pd.DataFrame:
    out = df.copy()
    num_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c]) and c not in exclude]

    for c in num_cols:
        col = out[c].astype(float)
        mn, mx = np.nanmin(col), np.nanmax(col)

        # Si la columna ya está en rango 0..100 (aprox), la dejamos tal cual
        if np.isfinite(mn) and np.isfinite(mx) and mn >= 0 and mx <= 100:
            continue

        # Si no hay variación, set a 0
        if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
            out[c] = 0.0
            continue

        out[c] = (col - mn) / (mx - mn) * 100.0

    return out

def add_derived_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def mean_if_exists(cols: list[str], new_name: str):
        present = [c for c in cols if c in out.columns]
        if len(present) == 0:
            return
        out[new_name] = out[present].mean(axis=1)

    mean_if_exists(PHYSICAL_COLS, "score_fisico")
    mean_if_exists(TECHNICAL_COLS, "score_tecnico")
    mean_if_exists(MENTAL_COLS, "score_mental")

    return out

# -----------------------------
# Main
# -----------------------------
def main():
    in_path = Path(INPUT_DB)
    if not in_path.exists():
        raise FileNotFoundError(f"No encuentro {INPUT_DB} en: {Path.cwd()}")

    conn = sqlite3.connect(INPUT_DB)
    tables = list_tables(conn)
    if not tables:
        raise RuntimeError("No se encontraron tablas en la base SQLite.")

    main_table = pick_main_table(tables)
    print(f"[INFO] Tablas encontradas: {tables}")
    print(f"[INFO] Usando tabla principal: {main_table}")

    df = pd.read_sql_query(f"SELECT * FROM {main_table};", conn)
    print(f"[INFO] Shape inicial: {df.shape}")

    # Si existe Player (para edad), hacemos join
    if "Player" in tables and "player_api_id" in df.columns:
        player = pd.read_sql_query("SELECT * FROM Player;", conn)

        # columnas típicas: player_api_id, birthday
        if "birthday" in player.columns:
            df = df.merge(
                player[["player_api_id", "birthday"]],
                on="player_api_id",
                how="left"
            )
            if "date" in df.columns:
                df["date_dt"] = safe_to_datetime(df["date"])
                df["birthday_dt"] = safe_to_datetime(df["birthday"])
                df["edad_estimada"] = (df["date_dt"] - df["birthday_dt"]).dt.days / 365.25
                # limpia auxiliares (opcional)
                df.drop(columns=["date_dt", "birthday_dt"], inplace=True, errors="ignore")
            else:
                # fallback: edad no se puede calcular sin fecha de registro
                print("[WARN] No existe columna 'date' para calcular edad por registro.")
        else:
            print("[WARN] Tabla Player existe pero no tiene 'birthday'.")

    # 1) Detectar posición (para medianas por posición)
    pos_col = find_position_col(df)
    print(f"[INFO] Columna de posición detectada: {pos_col}")

    # Si 'preferred_positions' viene como string con varias posiciones,
    # nos quedamos con la primera como "posición principal"
    if pos_col == "preferred_positions":
        df["posicion_principal"] = df["preferred_positions"].astype(str).str.split().str[0]
        pos_col = "posicion_principal"
        print("[INFO] Usando posicion_principal (primera de preferred_positions) para agrupar medianas.")

    # 2) Relleno de nulos por mediana (numéricas) / moda (categóricas) por posición
    df = fill_nulls_by_group_median(df, pos_col)

    # 3) Variables derivadas (scores)
    df = add_derived_scores(df)

    # 4) One-hot (categóricas)
    #    Excluimos IDs, y dejamos fuera columnas fecha completas (si existen) para no explotar cardinalidad
    exclude_ohe = set(ID_COLS)
    if "date" in df.columns:
        exclude_ohe.add("date")
    if "birthday" in df.columns:
        exclude_ohe.add("birthday")

    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in exclude_ohe]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dummy_na=False)

    # 5) Normalización 0-100 (numéricas)
    #    Excluimos IDs y (si quieres) el target overall_rating si lo vas a predecir sin escalar
    exclude_norm = set(ID_COLS)
    # Si quieres NO normalizar overall_rating, descomenta:
    # exclude_norm.add("overall_rating")

    df = minmax_0_100(df, exclude=exclude_norm)

    print(f"[INFO] Shape final (post OHE): {df.shape}")

    # 6) Guardar a nuevo SQLite
    out_conn = sqlite3.connect(OUTPUT_DB)
    df.to_sql(OUTPUT_TABLE, out_conn, if_exists="replace", index=False)

    out_conn.close()
    conn.close()

    print(f"[OK] Guardado: {OUTPUT_DB} (tabla: {OUTPUT_TABLE})")

if __name__ == "__main__":
    main()
