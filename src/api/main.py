import shap
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

MODEL_PATH = Path("models/xgboost_fraud.json")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
explainer = shap.TreeExplainer(model)

le = LabelEncoder()
le.fit(["grocery", "restaurant", "gas", "online", "travel"])

app = FastAPI(title="Fraud Detection API", version="1.0")

class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Сума транзакції в USD")
    hour: int = Field(..., ge=0, le=23)
    is_new_location: int = Field(..., ge=0, le=1)
    merchant_category: str = Field(..., description="grocery/restaurant/gas/online/travel")
    user_avg_amount: float = Field(..., gt=0)
    transactions_last_hour: int = Field(..., ge=0)
    is_foreign_country: int = Field(..., ge=0, le=1)
    card_present: int = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 2400.0,
                "hour": 3,
                "is_new_location": 1,
                "merchant_category": "online",
                "user_avg_amount": 45.0,
                "transactions_last_hour": 5,
                "is_foreign_country": 1,
                "card_present": 0,
            }
        }
    } 

class FraudPrediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str                    # LOW / MEDIUM / HIGH
    top_risk_factors: list[dict]       # топ-3 фічі що вплинули найбільше


def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.7:
        return "MEDIUM"
    return "HIGH"

@app.post("/predict", response_model=FraudPrediction)
def predict(transaction: TransactionRequest):
    try:
        cat_encoded = le.transform([transaction.merchant_category])[0]
        features = pd.DataFrame([{
            "amount":                  transaction.amount,
            "hour":                    transaction.hour,
            "is_new_location":         transaction.is_new_location,
            "merchant_category":       cat_encoded,
            "user_avg_amount":         transaction.user_avg_amount,
            "transactions_last_hour":  transaction.transactions_last_hour,
            "is_foreign_country":      transaction.is_foreign_country,
            "card_present":            transaction.card_present,
            "amount_vs_avg_ratio":     transaction.amount / (transaction.user_avg_amount + 1e-9),
        }])

        fraud_prob = float(model.predict_proba(features)[0][1])

        # 3. SHAP для цієї конкретної транзакції
        shap_vals = explainer.shap_values(features)[0]
        shap_series = pd.Series(shap_vals, index=features.columns)

        # топ-3 фічі за абсолютним SHAP значенням
        top_factors = (
            shap_series.abs()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )
        risk_factors = [
            {
                "feature": f,
                "value": float(features[f].iloc[0]),
                "shap_impact": float(shap_series[f]),
            }
            for f in top_factors
        ]

        return FraudPrediction(
            fraud_probability=round(fraud_prob, 4),
            is_fraud=fraud_prob >= 0.5,
            risk_level=get_risk_level(fraud_prob),
            top_risk_factors=risk_factors,
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model": "xgboost_fraud_v1"}