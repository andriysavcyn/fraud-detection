import random
import requests
from langchain_core.tools import tool

FASTAPI_URL = "http://localhost:8000"

@tool
def get_transaction_risk(transaction_json: str) -> str:
    """Викликає ML модель для оцінки ризику шахрайства (predict fraud probability). Передайте сюди JSON рядок з даними транзакції."""
    import json
    try:
        data = json.loads(transaction_json)
        response = requests.post(f"{FASTAPI_URL}/predict", json=data, timeout=5)
        result = response.json()

        factors = result["top_risk_factors"]
        factors_text = "\n".join(
            f"  - {f['feature']}: значення={f['value']:.2f}, вплив={f['shap_impact']:.3f}"
            for f in factors
        )
        return (
            f"Fraud probability: {result['fraud_probability']:.1%}\n"
            f"Risk level: {result['risk_level']}\n"
            f"Is fraud: {result['is_fraud']}\n"
            f"Топ фактори ризику:\n{factors_text}"
        )
    except Exception as e:
        return f"Помилка виклику ML моделі: {e}"
    
@tool
def get_user_profile(user_id: str) -> str:
    """Повертає профіль клієнта за його user_id. Містить середній чек, звичні країни та історію фрод-флагів."""
    profiles = {
        "user_001": {
            "avg_amount": 45.0,
            "typical_hours": "9:00-21:00",
            "usual_countries": ["Ukraine", "Poland"],
            "total_transactions": 234,
            "previous_fraud_flags": 0,
            "account_age_days": 890,
        },
        "user_002": {
            "avg_amount": 1200.0,
            "typical_hours": "8:00-23:00",
            "usual_countries": ["Ukraine", "Germany", "USA"],
            "total_transactions": 45,
            "previous_fraud_flags": 1,
            "account_age_days": 120,
        },
    }

    profile = profiles.get(user_id, {
        "avg_amount": random.uniform(30, 200),
        "typical_hours": "8:00-22:00",
        "usual_countries": ["Ukraine"],
        "total_transactions": random.randint(10, 500),
        "previous_fraud_flags": 0,
        "account_age_days": random.randint(30, 1000),
    })

    return (
        f"Профіль юзера {user_id}:\n"
        f"  Середній чек: ${profile['avg_amount']:.0f}\n"
        f"  Активні години: {profile['typical_hours']}\n"
        f"  Звичні країни: {', '.join(profile['usual_countries'])}\n"
        f"  Всього транзакцій: {profile['total_transactions']}\n"
        f"  Попередніх fraud-флагів: {profile['previous_fraud_flags']}\n"
        f"  Вік акаунту: {profile['account_age_days']} днів"
    )

@tool 
def get_fraud_patterns(risk_factors: str) -> str:
    """Аналізує текстові фактори ризику або назви полів та повертає відомі патерни шахрайства (наприклад, velocity_fraud, account_takeover)."""
    patterns_db = {
        "card_not_present": (
            "Card-Not-Present (CNP) Fraud: транзакція без фізичної картки. "
            "Типово для онлайн-покупок. Зловмисник використовує вкрадені "
            "дані картки. Рекомендація: верифікація через 3DS."
        ),
        "account_takeover": (
            "Account Takeover (ATO): злам акаунту з нової локації. "
            "Ознаки: нова країна + незвична година + велика сума. "
            "Рекомендація: заморозити до підтвердження особи."
        ),
        "velocity_fraud": (
            "Velocity Fraud: багато транзакцій за короткий час. "
            "Зловмисник перевіряє чи картка активна малими сумами, "
            "потім робить велику покупку. Рекомендація: rate limiting."
        ),
        "geo_anomaly": (
            "Geographic Anomaly: транзакція з незвичної локації. "
            "Особливо підозріло якщо попередня транзакція була в іншій країні "
            "менше ніж 2 години тому (impossible travel)."
        ),
    }

    result = []
    factors_lower = risk_factors.lower()

    if "card_present" in factors_lower or "online" in factors_lower:
        result.append(patterns_db["card_not_present"])
    if "new_location" in factors_lower or "foreign" in factors_lower:
        result.append(patterns_db["account_takeover"])
        result.append(patterns_db["geo_anomaly"])
    if "transactions_last_hour" in factors_lower or "velocity" in factors_lower:
        result.append(patterns_db["velocity_fraud"])

    if not result:
        result.append("Загальний підвищений ризик без специфічного патерну. Рекомендується мануальна перевірка.")

    return "\n\n".join(result)
