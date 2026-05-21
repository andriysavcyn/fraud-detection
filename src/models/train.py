import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DATA_PATH = Path("data/transactions.csv")
MODEL_DIR = Path("models")
MLFLOW_DIR = Path("mlflow_runs")
TARGET = "is_fraud"
CAT_FEATURE = "merchant_category"

try:
    MODEL_DIR.mkdir(exist_ok=True)
    MLFLOW_DIR.mkdir(exist_ok=True)
    logger.debug("Target directories initialized successfully.")
except Exception as e:
    logger.critical(
        f"Critical failure initializing workspace directories: {e}", exc_info=True
    )
    sys.exit(1)

def load_and_prepare(path: Path):
    logger.info(f"Attempting to load dataset from: {path}")

    # 1. File Reading Exception Handling
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"Data source file was not found at location: {path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"The source file at {path} contains no data.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while reading CSV file: {e}", exc_info=True)
        raise

    # 2. Structural Data Validation
    if CAT_FEATURE not in df.columns:
        err = (
            f"Required categorical column '{CAT_FEATURE}' is missing from the dataset."
        )
        logger.error(err)
        raise KeyError(err)

    if TARGET not in df.columns:
        err = f"Target label column '{TARGET}' is missing from the dataset."
        logger.error(err)
        raise KeyError(err)

    # 3. Encoding and Feature Transformation Phase
    try:
        logger.info(f"Encoding categorical feature: {CAT_FEATURE}")
        le = LabelEncoder()
        df[CAT_FEATURE] = le.fit_transform(df[CAT_FEATURE].astype(str))

        feature_cols = [c for c in df.columns if c != TARGET]
        X = df[feature_cols]
        y = df[TARGET]

        logger.info(f"Dataset extracted successfully. Shape: {df.shape}")
        return X, y, le
    except Exception as e:
        logger.error(
            f"Failed during data transformation and encoding: {e}", exc_info=True
        )
        raise

def train():
    # 1. MLflow Tracking Engine Setup
    try:
        mlflow.set_tracking_uri(MLFLOW_DIR.resolve().as_uri())
        mlflow.set_experiment("fraud-detection")
        logger.info(f"MLflow experiment tracking initialized at: {MLFLOW_DIR}")
    except Exception as e:
        logger.error(
            f"Failed to set up MLflow tracking environment: {e}", exc_info=True
        )
        raise

    # 2. Data Ingestion
    X, y, le = load_and_prepare(DATA_PATH)

    # 3. Train-Test Split with Stratification Validation
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        logger.info(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    except ValueError as e:
        logger.error(
            f"Splitting failed. Ensure data is large enough and has valid target distribution: {e}",
            exc_info=True,
        )
        raise

    # 4. Defensive Target Imbalance Calculation
    fraud_count = int((y == 1).sum())
    legit_count = int((y == 0).sum())

    if fraud_count == 0:
        logger.warning(
            "Zero fraud cases detected in target column. Defaulting scale_pos_weight to 1."
        )
        scale_pos_weight = 1
    else:
        scale_pos_weight = legit_count // fraud_count

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "random_state": 42,
    }

    model = None
    explainer = None

    # 5. Model Execution and Tracking Block
    try:
        with mlflow.start_run() as run:
            logger.info(f"Active MLflow Run started. Run ID: {run.info.run_id}")
            mlflow.log_params(params)

            logger.info("Initiating XGBoost Classifier training sequence...")
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
            logger.info("Model training completed successfully.")

            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)

            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("pr_auc", pr_auc)

            # Structured metrics reporting via logs instead of prints
            logger.info(
                f"Performance Summary:\nROC-AUC : {roc_auc:.4f}\nPR-AUC  : {pr_auc:.4f}"
            )

            rep = classification_report(y_test, y_pred, target_names=["legit", "fraud"])
            logger.info(f"Classification Report Summary:\n{rep}")

            # 6. SHAP Interpretability Phase
            logger.info("Computing local SHAP feature importances...")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)

                mean_shap = pd.Series(
                    np.abs(shap_values).mean(axis=0), index=X.columns
                ).sort_values(ascending=False)

                logger.info(
                    f"Top-5 Feature Importances (SHAP):\n{mean_shap.head().to_string()}"
                )
                mlflow.log_dict(mean_shap.to_dict(), "shap_importance.json")
            except Exception as e:
                logger.error(
                    f"Non-critical failure computing SHAP values or artifact logging: {e}",
                    exc_info=True,
                )

            # 7. Model Serialization and Artifact Storage
            logger.info("Serializing model architecture and pipeline metadata...")
            try:
                mlflow.xgboost.log_model(model, "xgboost_model")

                local_model_path = MODEL_DIR / "xgboost_fraud.json"
                model.save_model(local_model_path)

                feature_cols = list(X.columns)
                feature_path = MODEL_DIR / "feature_cols.csv"
                pd.Series(feature_cols).to_csv(feature_path, index=False)

                logger.info(
                    f"Model and features successfully mapped locally to environment → {MODEL_DIR.resolve()}"
                )
            except (PermissionError, IOError) as file_err:
                logger.error(
                    f"Disk access/IO breakdown saving models to path: {file_err}",
                    exc_info=True,
                )
                raise
            except Exception as e:
                logger.error(f"Unexpected artifact logging failure: {e}", exc_info=True)
                raise

    except Exception as run_err:
        logger.critical(
            f"Pipeline execution broke down inside active MLflow run: {run_err}",
            exc_info=True,
        )
        raise

    return model, explainer, X_test

if __name__ == "__main__":
    try:
        train()
        logger.info("Machine Learning execution script finished successfully.")
    except Exception as main_err:
        logger.critical(
            f"Execution terminated due to unhandled pipeline failure: {main_err}"
        )
        sys.exit(1)
