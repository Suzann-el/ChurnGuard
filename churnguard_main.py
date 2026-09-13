"""
ChurnGuard API — Prédiction du churn clients télécom
Modèle : LightGBM | Dataset : IBM Telco Customer Churn

Lancer : uvicorn churnguard_main:app --reload --port 8001
Docs   : http://localhost:8001/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import shap

# ── Chargement du modèle au démarrage ────────────────────────────────────────
app = FastAPI(
    title="ChurnGuard API",
    description="Prédiction du risque de churn clients télécom — LightGBM",
    version="1.0.0"
)

try:
    model         = joblib.load("churnguard_model.pkl")
    feature_names = joblib.load("churnguard_features.pkl")
    with open("churnguard_config.json") as f:
        config = json.load(f)
    THRESHOLD = config["threshold"]
    explainer = shap.TreeExplainer(model)
    print(f"Modele charge — AUC-ROC : {config['auc_roc']} | Seuil : {THRESHOLD}")
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modele : {e}")


# ── Schéma client ─────────────────────────────────────────────────────────────
class ClientProfile(BaseModel):
    gender:           str   = Field(..., example="Male")
    SeniorCitizen:    int   = Field(..., ge=0, le=1, example=0)
    Partner:          str   = Field(..., example="Yes")
    Dependents:       str   = Field(..., example="No")
    tenure:           int   = Field(..., ge=0, example=12)
    PhoneService:     str   = Field(..., example="Yes")
    MultipleLines:    str   = Field(..., example="No")
    InternetService:  str   = Field(..., example="Fiber optic")
    OnlineSecurity:   str   = Field(..., example="No")
    OnlineBackup:     str   = Field(..., example="No")
    DeviceProtection: str   = Field(..., example="No")
    TechSupport:      str   = Field(..., example="No")
    StreamingTV:      str   = Field(..., example="Yes")
    StreamingMovies:  str   = Field(..., example="Yes")
    Contract:         str   = Field(..., example="Month-to-month")
    PaperlessBilling: str   = Field(..., example="Yes")
    PaymentMethod:    str   = Field(..., example="Electronic check")
    MonthlyCharges:   float = Field(..., example=79.85)
    TotalCharges:     float = Field(..., example=958.2)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(client: ClientProfile) -> pd.DataFrame:
    data = pd.DataFrame([client.dict()])

    # Encodage identique au notebook
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        if data[col].nunique() <= 2:
            data[col] = le.fit_transform(data[col].astype(str))
        else:
            dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
            data = pd.concat([data.drop(columns=[col]), dummies], axis=1)

    # Feature engineering
    data['charges_per_tenure'] = np.where(
        data['tenure'] > 0,
        data['TotalCharges'] / data['tenure'],
        data['MonthlyCharges']
    )
    service_cols = [c for c in data.columns if any(s in c for s in
        ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
         'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])]
    data['nb_services']      = data[service_cols].sum(axis=1) if service_cols else 0
    data['risque_financier'] = int(
        (data['MonthlyCharges'].values[0] > 65) and (data['tenure'].values[0] < 12))
    data['tenure_ratio']     = data['tenure'] / 72

    # Aligner avec les features du modele
    for col in feature_names:
        if col not in data.columns:
            data[col] = 0
    return data[feature_names].astype(float)


def get_risk_level(proba: float) -> str:
    if proba >= 0.70: return "CRITIQUE"
    if proba >= 0.50: return "ELEVE"
    if proba >= 0.30: return "MODERE"
    return "FAIBLE"


def get_recommendation(risk_level: str) -> str:
    reco = {
        "CRITIQUE": "Contact immediat — offre de retention prioritaire (remise, upgrade)",
        "ELEVE":    "Contact sous 48h — proposer un changement de contrat annuel",
        "MODERE":   "Surveiller — envoyer une offre de fidelisation par email",
        "FAIBLE":   "Client fidele — pas d'action requise",
    }
    return reco.get(risk_level, "Suivi standard")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":        "ChurnGuard API",
        "modele":         "LightGBM",
        "version":        config["model_version"],
        "auc_roc":        config["auc_roc"],
        "seuil_decision": THRESHOLD,
        "documentation":  "/docs"
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": model is not None,
        "n_features":   len(feature_names)
    }


@app.post("/predict")
def predict(client: ClientProfile):
    """
    Prédit le risque de churn pour un client.

    Retourne :
    - probabilite_churn : score entre 0 et 1
    - churn_predit : 1 si churn probable, 0 sinon
    - niveau_risque : FAIBLE / MODERE / ELEVE / CRITIQUE
    - top_3_facteurs : facteurs les plus influents (SHAP)
    - recommandation : action commerciale recommandee
    """
    try:
        X = preprocess(client)

        proba      = float(model.predict_proba(X)[0, 1])
        churn      = int(proba >= THRESHOLD)
        risk_level = get_risk_level(proba)

        # SHAP
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv_arr = sv[1]
        elif hasattr(sv, 'ndim') and sv.ndim == 3:
            sv_arr = sv[:, :, 1]
        else:
            sv_arr = sv

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "impact":  sv_arr[0],
            "valeur":  X.values[0]
        }).sort_values("impact", key=abs, ascending=False)

        top_3 = [
            {
                "feature":   row["feature"],
                "impact":    round(float(row["impact"]), 4),
                "direction": "augmente le risque" if row["impact"] > 0 else "diminue le risque",
                "valeur":    round(float(row["valeur"]), 2)
            }
            for _, row in shap_df.head(3).iterrows()
        ]

        return {
            "probabilite_churn": round(proba, 4),
            "churn_predit":      churn,
            "niveau_risque":     risk_level,
            "seuil_utilise":     THRESHOLD,
            "top_3_facteurs":    top_3,
            "recommandation":    get_recommendation(risk_level)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(clients: list[ClientProfile]):
    """Prédit le churn pour une liste de clients."""
    if len(clients) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 clients par requete.")

    results = []
    for i, c in enumerate(clients):
        try:
            result = predict(c)
            result["client_index"] = i
            results.append(result)
        except Exception as e:
            results.append({"client_index": i, "erreur": str(e)})

    n_churn = sum(r.get("churn_predit", 0) for r in results)
    return {
        "total_clients":   len(results),
        "churns_predits":  n_churn,
        "taux_churn_predit": round(n_churn / len(results) * 100, 1),
        "predictions":     results
    }
