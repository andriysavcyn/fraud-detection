import numpy as np
import pandas as pd
from pathlib import Path

def generate_transactions(n_samples: int=10000, random_state: int=42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_fraud = int(n_samples * 0.05)
    n_legit = n_samples - n_fraud

    legit = pd.DataFrame({
        "amount": rng.lognormal(mean=3.5, sigma=0.8, size=n_legit),
        "hour": rng.integers(8, 22, size=n_legit),
        "is_new_location": rng.choice([0, 1], size=n_legit, p=[0.95, 0.05]),
        "merchant_category": rng.choice(
            ["grocery", "restaurant", "gas", "online", "travel"],
            size=n_legit, p=[0.3, 0.25, 0.2, 0.15, 0.1]
        ),
        "user_avg_amount": rng.lognormal(mean=3.5, sigma=0.5, size=n_legit),
        "transactions_last_hour": rng.integers(0, 3, size=n_legit),
        "is_foreign_country": rng.choice([0, 1], size=n_legit, p=[0.92, 0.08]),
        "card_present": rng.choice([0, 1], size=n_legit, p=[0.2, 0.8]),
        "is_fraud": 0,
    })

    fraud = pd.DataFrame({
        "amount": rng.lognormal(mean=5.5, sigma=1.2, size=n_fraud),  # ~$245, більші суми
        "hour": rng.choice(
            list(range(0, 5)) + list(range(22, 24)),
            size=n_fraud                                               # вночі
        ),
        "is_new_location": rng.choice([0, 1], size=n_fraud, p=[0.2, 0.8]),
        "merchant_category": rng.choice(
            ["grocery", "restaurant", "gas", "online", "travel"],
            size=n_fraud, p=[0.05, 0.05, 0.05, 0.6, 0.25]            # online/travel домінують
        ),
        "user_avg_amount": rng.lognormal(mean=3.5, sigma=0.5, size=n_fraud),
        "transactions_last_hour": rng.integers(3, 10, size=n_fraud),  # багато транзакцій
        "is_foreign_country": rng.choice([0, 1], size=n_fraud, p=[0.3, 0.7]),
        "card_present": rng.choice([0, 1], size=n_fraud, p=[0.75, 0.25]),  # CNP fraud
        "is_fraud": 1,
    })

    df = pd.concat([legit, fraud], ignore_index=True)

    df["amount_vs_avg_ratio"] = df["amount"] / (df["user_avg_amount"] + 1e-9)
    
    return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

if __name__ == "__main__":
    df = generate_transactions()
    output_path = Path("data/transactions.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} transactions → {output_path}")
    print(f"Fraud rate: {df['is_fraud'].mean():.1%}")
    print(df.head())