"""Batch simulate players using ML outputs as inputs (placeholder friendly).

Genera simulaciones para 5 jugadores reales seleccionados del dataset
`player_attributes_clean` y guarda CSVs + un resumen JSON con recomendaciones.

Run: python EDO___sIMULACION/batch_simulate_players.py
"""
import sqlite3
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd

from EDO___sIMULACION import edo_core
from models import build_position_target_by_rules


DB = Path(__file__).resolve().parent.parent / 'database_clean.sqlite'
OUTDIR = Path(__file__).resolve().parent / 'sim_outputs'
OUTDIR.mkdir(exist_ok=True)


def load_regression_model(pkl_path: Path):
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, 'rb') as f:
            m = pickle.load(f)
        return m
    except Exception:
        return None


def predict_potential(model, df_row):
    # model could be a scikit-learn-like object or custom; try common interfaces
    if model is None:
        return None
    try:
        # expect model to accept 2D array
        X = np.asarray(df_row).reshape(1, -1)
        if hasattr(model, 'predict'):
            return float(model.predict(X).reshape(-1)[0])
        if hasattr(model, 'predecir'):
            return float(model.predecir(X)[0])
    except Exception:
        return None


def map_label_to_pos(label: str) -> str:
    lab = label.lower()
    if 'portero' in lab or 'gk' in lab:
        return 'GK'
    if 'defen' in lab or 'defensa' in lab or 'central' in lab:
        return 'DEF'
    if 'medio' in lab or 'centro' in lab or 'mid' in lab:
        return 'MID'
    return 'FWD'


def main():
    if not DB.exists():
        raise FileNotFoundError(f"No database found at {DB}")

    conn = sqlite3.connect(str(DB))
    df = pd.read_sql_query('SELECT * FROM player_attributes_clean;', conn)
    conn.close()

    # pick 5 players across quantiles of overall_rating
    if 'overall_rating' in df.columns:
        qs = [0.05, 0.275, 0.5, 0.725, 0.95]
        vals = df['overall_rating'].quantile(qs).values
        players = []
        for v in vals:
            # pick nearest
            idx = (df['overall_rating'] - v).abs().idxmin()
            players.append(df.loc[idx])
        players_df = pd.DataFrame(players).drop_duplicates(subset=['player_api_id']).reset_index(drop=True)
    else:
        players_df = df.sample(5, random_state=42).reset_index(drop=True)

    # try load regression model pickle if exists
    pkl = Path(__file__).resolve().parent.parent / 'red1_regresion_trained.pkl'
    reg_model = load_regression_model(pkl)

    summaries = []
    for i, row in players_df.iterrows():
        pid = row.get('player_api_id', f'player_{i}')
        name = row.get('player_name', f'player_{pid}') if 'player_name' in row else f'player_{pid}'

        # obtain potential prediction: prefer model, then 'potential' column, else overall_rating
        potential = None
        if 'potential' in row and not pd.isna(row['potential']):
            potential = float(row['potential'])

        if potential is None and reg_model is not None:
            # attempt to use top features if stored in metadata file
            try:
                meta = json.load(open(Path(__file__).resolve().parent.parent / 'estudiante_a_red1_results.json', 'r'))
                feats = meta.get('features', [])
                X = []
                for f in feats:
                    X.append(float(row.get(f, 50.0)))
                pred = predict_potential(reg_model, X)
                if pred is not None:
                    potential = float(pred)
            except Exception:
                potential = None

        if potential is None:
            potential = float(row.get('overall_rating', 50.0))

        # position via rules
        try:
            y, _, _, labels = build_position_target_by_rules(pd.DataFrame([row]))
            pos_label = y[0]
        except Exception:
            pos_label = 'Medio'

        pos = map_label_to_pos(pos_label)

        # calibrate params
        params, weights = edo_core.calibrate_params_from_ml(potential, pos)

        # simple training regime defaults depending on scenario
        train = edo_core.TrainingRegime(EF=0.7, ET=0.7, EM=0.6)

        # define injury scenarios for variety
        injuries = []
        if i == 1:
            injuries = [edo_core.InjuryEvent(start_year=1.5, duration_years=0.5, severity=0.4, mode='exp_recovery')]
        elif i == 3:
            injuries = [edo_core.InjuryEvent(start_year=3.0, duration_years=1.0, severity=0.65, mode='shock')]

        fatigue_cfg = edo_core.FatigueConfig(enabled=True, k=0.12, recovery=0.25, cap=1.0)

        y0 = (float(row.get('score_fisico', row.get('overall_rating', 50.0))) if 'score_fisico' in row else float(row.get('overall_rating', 50.0)),
              float(row.get('score_tecnico', 50.0)) if 'score_tecnico' in row else float(row.get('overall_rating', 50.0)),
              float(row.get('score_mental', 50.0)) if 'score_mental' in row else float(row.get('overall_rating', 50.0)))

        sim = edo_core.simulate_player(years=10.0, dt=1/12, age0=float(row.get('age', 18.0)), y0=y0,
                                       params=params, weights=weights, train_regime=train,
                                       injuries=injuries, fatigue_cfg=fatigue_cfg)

        out_csv = OUTDIR / f'sim_player_{pid}.csv'
        pd.DataFrame(sim).to_csv(out_csv, index=False)

        summary = {
            'player_api_id': int(pid) if isinstance(pid, (int, np.integer)) else str(pid),
            'player_name': name,
            'age0': float(row.get('age', 18.0)),
            'position': pos,
            'potential_used': float(potential),
            'output_csv': str(out_csv),
            'final_R': float(sim['R'][-1]),
            'max_R': float(np.max(sim['R']))
        }
        summaries.append(summary)

    # save summaries
    with open(OUTDIR / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"Simulaciones guardadas en {OUTDIR}")


if __name__ == '__main__':
    main()
