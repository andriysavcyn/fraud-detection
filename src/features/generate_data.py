import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def generate_transactions(
    n_samples: int = 10000, random_state: int = 42
) -> pd.DataFrame:
    # Validation of input data
    if not isinstance(n_samples, int) or n_samples <= 0:
        error_msg = f"n_samples must be a positive integer, resulting in: {n_samples}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(
        f"Start of transaction generation (n_samples={n_samples}, random_state={random_state})..."
    )

    try:
        rng = np.random.default_rng(random_state)
        n_fraud = int(n_samples * 0.05)
        n_legit = n_samples - n_fraud

        logger.debug(f"Generation of {n_legit} legitimate transactions...")
        legit = pd.DataFrame(
            {
                "amount": rng.lognormal(mean=3.5, sigma=0.8, size=n_legit),
                "hour": rng.integers(8, 22, size=n_legit),
                "is_new_location": rng.choice([0, 1], size=n_legit, p=[0.95, 0.05]),
                "merchant_category": rng.choice(
                    ["grocery", "restaurant", "gas", "online", "travel"],
                    size=n_legit,
                    p=[0.3, 0.25, 0.2, 0.15, 0.1],
                ),
                "user_avg_amount": rng.lognormal(mean=3.5, sigma=0.5, size=n_legit),
                "transactions_last_hour": rng.integers(0, 3, size=n_legit),
                "is_foreign_country": rng.choice([0, 1], size=n_legit, p=[0.92, 0.08]),
                "card_present": rng.choice([0, 1], size=n_legit, p=[0.2, 0.8]),
                "is_fraud": 0,
            }
        )

        logger.debug(f"Generation of {n_legit} fraud transactions...")
        fraud = pd.DataFrame(
            {
                "amount": rng.lognormal(mean=5.5, sigma=1.2, size=n_fraud),
                "hour": rng.choice(
                    list(range(0, 5)) + list(range(22, 24)), size=n_fraud
                ),
                "is_new_location": rng.choice([0, 1], size=n_fraud, p=[0.2, 0.8]),
                "merchant_category": rng.choice(
                    ["grocery", "restaurant", "gas", "online", "travel"],
                    size=n_fraud,
                    p=[0.05, 0.05, 0.05, 0.6, 0.25],
                ),
                "user_avg_amount": rng.lognormal(mean=3.5, sigma=0.5, size=n_fraud),
                "transactions_last_hour": rng.integers(3, 10, size=n_fraud),
                "is_foreign_country": rng.choice([0, 1], size=n_fraud, p=[0.3, 0.7]),
                "card_present": rng.choice([0, 1], size=n_fraud, p=[0.75, 0.25]),
                "is_fraud": 1,
            }
        )

        df = pd.concat([legit, fraud], ignore_index=True)
        df["amount_vs_avg_ratio"] = df["amount"] / (df["user_avg_amount"] + 1e-9)

        logger.info("Successful data merging and shuffling!")
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    except MemoryError:
        logger.error(
            "There is not enough memory to generate this many transactions.",
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data generation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    output_path = Path("data/transactions.csv")

    try:
        df = generate_transactions()

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(
                f"Failed to create directory {output_path.parent}: {e}", exc_info=True
            )
            sys.exit(1)

        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(df)} transaction → {output_path}")
            logger.info(f"Fraud rate: {df['is_fraud'].mean():.1%}")
            logger.info(f"Sample data:\n{df.head()}")
        except PermissionError:
            logger.error(
                f"Access denied when trying to write to file: {output_path}",
                exc_info=True,
            )
        except IOError as e:
            logger.error(f"I/O error while saving file: {e}", exc_info=True)

    except Exception as e:
        logger.critical(f"Critical error in program execution: {e}", exc_info=True)