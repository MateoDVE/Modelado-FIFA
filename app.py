import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import sys
import os

# Ensure models.py can be imported
sys.path.append(os.getcwd())
from models import MLPRegressor, MLPClassifier, ID_COLS, rmse, mae, r2_score, confusion_matrix
from EDO___sIMULACION.edo_core import (
    simulate_player, calibrate_params_from_ml, TrainingRegime, InjuryEvent, WeightsR
)

app = Flask(__name__)

# Load models
red1_model = None
red1_features = []
red2_model = None
red2_features = []
red2_classes = []

def load_models():
    global red1_model, red1_features, red2_model, red2_features, red2_classes
    
    try:
        with open("red1_regresion_trained.pkl", "rb") as f:
            data = pickle.load(f)
            red1_model = data['model']
            red1_features = data['features']
            print("Red 1 loaded successfully.")
    except Exception as e:
        print(f"Error loading Red 1: {e}")

    try:
        with open("red2_clasificacion_trained.pkl", "rb") as f:
            data = pickle.load(f)
            red2_model = data['model']
            red2_features = data['features']
            red2_classes = data['classes']
            print("Red 2 loaded successfully.")
    except Exception as e:
        print(f"Error loading Red 2: {e}")

load_models()

# Feature mapping
FEATURE_GROUPS = {
    "Fisico": ["score_fisico", "acceleration", "sprint_speed", "stamina", "strength", "agility", "shot_power", "jumping", "balance"],
    "Tecnica": ["score_tecnico", "ball_control", "dribbling", "short_passing", "long_passing", "crossing", "curve", "finishing", "volleys", "penalties", "long_shots", "free_kick_accuracy", "heading_accuracy"],
    "Mentalidad": ["score_mental", "reactions", "vision", "positioning", "interceptions", "marking", "standing_tackle", "sliding_tackle", "aggression"]
}

def map_inputs_to_features(inputs, feature_list):
    """
    Maps high-level inputs (Fisico, Tecnica, Mentalidad) to detailed features.
    """
    row = {}
    
    # Default values for features not covered by inputs (if any)
    # We'll use the average of the inputs as a fallback or 50 (neutral)
    default_val = (inputs.get("Fisico", 50) + inputs.get("Tecnica", 50) + inputs.get("Mentalidad", 50)) / 3
    
    for feature in feature_list:
        assigned = False
        for group, feats in FEATURE_GROUPS.items():
            if feature in feats:
                row[feature] = inputs.get(group, 50)
                assigned = True
                break
        
        if not assigned:
            # If feature is not in our groups, try to assign based on name similarity or fallback
            if "passing" in feature:
                row[feature] = inputs.get("Tecnica", 50)
            elif "shot" in feature or "finish" in feature:
                row[feature] = inputs.get("Tecnica", 50)
            elif "defens" in feature or "tackle" in feature or "mark" in feature:
                row[feature] = inputs.get("Mentalidad", 50) # Defense often grouped with mentality/tactical in simple models, or maybe physical? Let's stick to Mentalidad for tactical/defensive awareness if not specified.
            else:
                row[feature] = default_val
                
    return pd.DataFrame([row])

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Inputs from UI
    fisico = float(data.get('Fisico', 50))
    tecnica = float(data.get('Tecnica', 50))
    mentalidad = float(data.get('Mentalidad', 50))
    rating = float(data.get('Ratting', 50))
    edad = float(data.get('Edad', 25))
    
    inputs = {
        "Fisico": fisico,
        "Tecnica": tecnica,
        "Mentalidad": mentalidad
    }
    
    response = {}
    
    # Red 1 Prediction (Potential)
    if red1_model:
        X_red1 = map_inputs_to_features(inputs, red1_features)
        # Ensure column order matches training
        X_red1 = X_red1[red1_features]
        
        # Normalize if the model expects normalized data?
        # The training script likely normalized data. 
        # However, without the scaler object, we might be in trouble if the model expects scaled data (e.g. 0-1 or z-score).
        # Let's check models.py or estudiante_a_entrenamiento.py for scaling.
        # MLPRegressor usually requires scaling.
        
        pred_potential = red1_model.predict(X_red1)[0]
        response['potential'] = float(pred_potential)
        response['growth'] = float(pred_potential) - rating
    
    # Red 2 Prediction (Position)
    if red2_model:
        X_red2 = map_inputs_to_features(inputs, red2_features)
        X_red2 = X_red2[red2_features]
        
        pred_idx = red2_model.predict(X_red2)[0]
        # If predict returns index or label?
        # MLPClassifier in models.py: predict returns class labels if encoded?
        # Let's assume it returns the class label directly or index.
        # If it returns index, we use red2_classes.
        
        # Checking models.py MLPClassifier.predict:
        # It computes forward pass, then argmax. So it returns an index (0 to K-1).
        # We need to map it back to class name using red2_classes.
        
        if isinstance(pred_idx, (int, np.integer)):
             pred_position = red2_classes[pred_idx]
        else:
             pred_position = pred_idx # In case it returns label directly (unlikely for custom MLP)
             
        response['position'] = str(pred_position)
        
        # Get probabilities if possible
        # MLPClassifier has predict_proba?
        if hasattr(red2_model, 'predict_proba'):
            probs = red2_model.predict_proba(X_red2)[0]
            response['position_probs'] = {str(c): float(p) for c, p in zip(red2_classes, probs)}

    # --- SIMULATION (EDO) ---
    # Inputs for simulation
    # Talento maps to Tecnica
    # Lesiones comes from UI (0 to 100, mapped to probability/severity)
    lesiones_val = float(data.get('Lesiones', 0))
    
    # Setup simulation
    y0 = (fisico, tecnica, mentalidad)
    age0 = edad
    
    # Calibrate params (using a default potential or the predicted one if available)
    potential_sim = response.get('potential', 85.0)
    position_sim = response.get('position', 'DEFAULT')
    
    params, w = calibrate_params_from_ml(potential_pred=potential_sim, position=position_sim)
    
    # --- ADJUST PARAMS BASED ON INPUTS (New Logic) ---
    # 1. Fisico: High initial physical stats -> steeper decline after 30 (burnout/wear)
    #    Low initial physical -> flatter decline (less wear)
    #    Base slopeF is 0.10.
    if fisico > 80:
        params.slopeF = 0.15
    elif fisico < 50:
        params.slopeF = 0.08
        
    # 2. Tecnica: High technique -> slower decline (aging gracefully)
    #    Base slopeT is 0.08.
    if tecnica > 80:
        params.slopeT = 0.04  # Very slow decline
    elif tecnica < 50:
        params.slopeT = 0.10  # Faster decline if no technique to rely on
        
    # 3. Lesiones: Increases decay rates significantly
    #    Map 0-100 to a multiplier or additive factor
    risk_factor = lesiones_val / 100.0
    
    # Increase slopes based on injury risk
    params.slopeF += 0.20 * risk_factor  # Injuries hit physical hard
    params.slopeT += 0.05 * risk_factor  # Technique affected slightly
    params.slopeM += 0.02 * risk_factor  # Mentality affected slightly
    
    # Also increase base decay (beta0) slightly with high injury risk
    params.betaF0 += 0.05 * risk_factor
    
    # Training regime (default moderate-high)
    train_regime = TrainingRegime(EF=0.7, ET=0.7, EM=0.7)
    
    # Injuries
    injuries = []
    if lesiones_val > 0:
        # Map 0-100 slider to severity/probability
        # Let's say if > 0, we schedule a random injury or a fixed one scaled by severity
        # Severity 0.0 to 0.8 (max injury)
        severity = (lesiones_val / 100.0) * 0.8
        if severity > 0.05:
            # Add an injury at year 2
            injuries.append(InjuryEvent(start_year=2.0, duration_years=0.5, severity=severity, mode="shock"))
            
    # Run simulation until age 50 (or at least 5 years)
    target_age = 50.0
    sim_years = max(5.0, target_age - age0)
    
    sim = simulate_player(
        years=sim_years, 
        dt=1/52, 
        age0=age0, 
        y0=y0, 
        params=params, 
        weights=w, 
        train_regime=train_regime, 
        injuries=injuries
    )
    
    # Downsample for chart (every ~4 weeks)
    step = 4
    response['simulation'] = {
        't': [round(t, 2) for t in sim['t'][::step]],
        'age': [round(a, 1) for a in sim['age'][::step]],
        'R': [round(r, 1) for r in sim['R'][::step]],
        'F': [round(f, 1) for f in sim['F'][::step]],
        'T': [round(t, 1) for t in sim['T'][::step]],
        'M': [round(m, 1) for m in sim['M'][::step]]
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
