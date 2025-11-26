"""
pipeline.py
Main pipeline to load data, prepare it, train multiple models, evaluate them, and save results.
Steps:
1. Load data from CSV.
2. Split into train/test, scale features.
3. Train models: XGBoost, Logistic Regression, Random Forest, SVM, LightGBM, CatBoost.
4. Save trained models with unique versioned names.
5. Evaluate models: confusion matrix, probability plots, performance metrics.
Each step is wrapped with run_step for logging and error handling.
"""

import os
import sys
import json
import joblib
import logging
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from develop_model import DevelopWeatherModel
from trainers import (
    TrainingManager, XGBTrainer, LogRegTrainer,
    RFTrainer, SVMTrainer, LGBMTrainer, CatBoostTrainer
)
from evaluators import (
    EvaluationManager, XGBEvaluator, LogRegEvaluator,
    RFEvaluator, SVMEvaluator, LGBMEvaluator, CatBoostEvaluator
)

# -------------------------------------------------------------------------
# Config dataclass
# -------------------------------------------------------------------------
@dataclass
class Config:
    input_csv: str
    output_dir: str
    log_dir: str
    target: str
    brand_weight: float
    non_brand_weight: float
    features: list
    random_state: int
    test_size: float
    run_models: dict
    xgb_params: dict
    rf_params: dict
    logreg_params: dict
    svm_params: dict
    lgbm_params: dict
    catboost_params: dict
    scale_pos_weight: float = 1.0

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "pipeline.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    # File handler
    fh = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=5)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logger.handlers.clear()
    logger.addHandler(sh)
    logger.addHandler(fh)


# -------------------------------------------------------------------------
# run_step utility
# -------------------------------------------------------------------------
import time
class StepTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        logging.info(f"Start: {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        if exc_type:
            logging.error(f"Failed: {self.name} after {dt:.2f}s | {exc_type}", exc_info=True)
        else:
            logging.info(f"Done: {self.name} in {dt:.2f}s")
        return False

def run_step(name: str, fn, *args, **kwargs):
    with StepTimer(name):
        try:
            out = fn(*args, **kwargs)
            return True, out
        except Exception:
            logging.exception(f"{name} raised an exception")
            return False, None


# -------------------------------------------------------------------------
# Pipeline class
# -------------------------------------------------------------------------
class WeatherPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dm = DevelopWeatherModel()
        self.scaler: StandardScaler = None
        self.models: Dict[str, Any] = {}

    # ------------------------ Data preparation --------------------------
    def prepare_data(self, df: pd.DataFrame):
        df = df.dropna().copy()
        df["sample_weight"] = df[self.cfg.target].apply(
            lambda x: self.cfg.brand_weight if x == 1 else self.cfg.non_brand_weight
        )

        ALL_FEATURES = self.cfg.features + ["STN"]
        NUMERIC_FEATURES = self.cfg.features
        CAT_FEATURES = ["STN"]

        x = df[ALL_FEATURES]
        y = df[self.cfg.target]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.cfg.test_size, random_state=self.cfg.random_state
        )

        w_train = df.loc[x_train.index, "sample_weight"]

        # Compute scale_pos_weight
        self.cfg.scale_pos_weight = len(df[df[self.cfg.target] == 0]) / max(1, len(df[df[self.cfg.target] == 1]))

        # Scale numeric features
        self.scaler = StandardScaler()
        self.scaler.fit(x_train[NUMERIC_FEATURES])

        x_train_scaled = pd.DataFrame(
            self.scaler.transform(x_train[NUMERIC_FEATURES]),
            index=x_train.index, columns=NUMERIC_FEATURES
        )
        x_test_scaled = pd.DataFrame(
            self.scaler.transform(x_test[NUMERIC_FEATURES]),
            index=x_test.index, columns=NUMERIC_FEATURES
        )

        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test.reset_index(drop=True),
            "w_train": w_train,
            "x_train_scaled": x_train_scaled,
            "x_test_scaled": x_test_scaled,
            "NUMERIC_FEATURES": NUMERIC_FEATURES,
            "ALL_FEATURES": ALL_FEATURES,
            "CAT_FEATURES": CAT_FEATURES
        }
    
    def save_models(self, models: Dict[str, Any], scaler: StandardScaler, output_dir: str) -> dict:
        """
        Slaat modellen op en geeft een dict terug met de originele model_type en de naam die is opgeslagen
        """
        os.makedirs(output_dir, exist_ok=True)
        models_to_save = {**models, "Scaler": scaler}
        saved_names = {}

        # Sla elk model op met een unieke naam
        for model_type, model in models_to_save.items():
            latest_model, latest_version = self.dm.find_latest_model(output_dir, model_type)
            model_name = self.dm.generate_new_model_name(latest_version, model_type)
            path = os.path.join(output_dir, model_name)
            joblib.dump(model, path)
            logging.info(f"Saved {model_type} to {path}")
            saved_names[model_type] = model_name
        return saved_names

    # ------------------------ Pipeline run ------------------------------
    def run(self):
        # 1. Load data
        success, df = run_step("Load Data", pd.read_csv, self.cfg.input_csv)
        if not success: 
            return
        logging.info(f"Loaded dataset shape: {df.shape}")

        # 2. Prepare data
        success, prepared_data = run_step("Prepare Data", self.prepare_data, df)
        if not success: 
            return

        # 3. Train models
        trainers = [
            XGBTrainer(self.cfg, self.dm, "xgboost"),
            LogRegTrainer(self.cfg, self.dm, "logistic_regression"),
            RFTrainer(self.cfg, self.dm, "random_forest"),
            SVMTrainer(self.cfg, self.dm, "svm"),
            LGBMTrainer(self.cfg, self.dm, "lightgbm"),
            CatBoostTrainer(self.cfg, self.dm, "catboost"),
        ]
        models_to_run = []
        for trainer in trainers:
            model_key = trainer.model_name
            if self.cfg.run_models.get(model_key, False):
                models_to_run.append(trainer)
        trainers = models_to_run
        train_mgr = TrainingManager(trainers)

        success, models = run_step(
            "Train Models",
            train_mgr.train_all,
            x_train_scaled = prepared_data["x_train_scaled"],
            x_train = prepared_data["x_train"],
            y_train = prepared_data["y_train"],
            w_train = prepared_data["w_train"],
            cat_features = prepared_data["CAT_FEATURES"],
            numeric_features = prepared_data["NUMERIC_FEATURES"],
            all_features = prepared_data["ALL_FEATURES"]
        )
        if not success: 
            return
        self.models = models

        # 4. Save model names
        succes, saved_names = run_step(
            "Save Models",
            self.save_models,
            models,
            self.scaler,
            self.cfg.output_dir
        )
        if not succes:
            return

        # 5. Evaluate models
        evaluators = [
            XGBEvaluator(self.cfg, self.dm, "xgboost"),
            LogRegEvaluator(self.cfg, self.dm, "logistic_regression"),
            RFEvaluator(self.cfg, self.dm, "random_forest"),
            SVMEvaluator(self.cfg, self.dm, "svm"),
            LGBMEvaluator(self.cfg, self.dm, "lightgbm"),
            CatBoostEvaluator(self.cfg, self.dm, "catboost"),
        ]
        model_to_evaluate = []
        for evaluator in evaluators:
            model_key = evaluator.model_name
            if self.cfg.run_models.get(model_key, False):
                model_to_evaluate.append(evaluator)
        evaluators = model_to_evaluate
        eval_mgr = EvaluationManager(evaluators, saved_names=saved_names)

        success, _ = run_step(
            "Evaluate Models",
            eval_mgr.evaluate_all,
            self.models,
            prepared_data["x_test_scaled"],
            prepared_data["x_test"],
            prepared_data["y_test"],
            prepared_data["x_test"],   # station_data
            prepared_data["ALL_FEATURES"],
            prepared_data["NUMERIC_FEATURES"],
            prepared_data["CAT_FEATURES"],
            self.cfg.output_dir,
        )
        if not success: 
            return

# -------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = Config.from_json("config.json")
    setup_logging(cfg.log_dir)
    pipeline = WeatherPipeline(cfg)
    pipeline.run()
