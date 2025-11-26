"""
trainers.py

Defines BaseTrainer and concrete trainers for different ML algorithms.
Each trainer wraps the corresponding method from DevelopWeatherModel,
and specifies which features (scaled/unscaled, categorical) should be used.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd


# -------------------------------------------------------------------------
# Abstract Base Trainer
# -------------------------------------------------------------------------
class BaseTrainer(ABC):
    def __init__(self, cfg: Any, dm: Any, model_name: str):
        """
        cfg: Config object
        dm: DevelopWeatherModel instance (with create_* methods)
        model_name: short identifier (e.g., "XGB", "logreg")
        """
        self.cfg = cfg
        self.dm = dm
        self.model_name = model_name

    @abstractmethod
    def train(self,
              x_train_scaled: pd.DataFrame,
              x_train: pd.DataFrame,
              y_train: pd.Series,
              w_train: pd.Series,
              cat_features: list,
              numeric_features: list,
              all_features: list,
              pos_weight: float = None) -> Any:
        """Train and return the model"""
        pass


# -------------------------------------------------------------------------
# Concrete Trainers
# -------------------------------------------------------------------------
class XGBTrainer(BaseTrainer):
    def train(self, x_train_scaled, x_train, y_train, w_train,
              cat_features, numeric_features, all_features, pos_weight=None):
        params = {**self.cfg.xgb_params, "scale_pos_weight": pos_weight}
        return self.dm.create_xgb_model(
            x_train[numeric_features],
            y_train,
            params,
            w_train
        )


class LogRegTrainer(BaseTrainer):
    def train(self, x_train_scaled, x_train, y_train, w_train,
              cat_features, numeric_features, all_features, pos_weight=None):
        return self.dm.create_logreg_model(
            x_train_scaled[numeric_features],
            y_train,
            self.cfg.logreg_params,
            w_train
        )


class RFTrainer(BaseTrainer):
    def train(self, x_train_scaled, x_train, y_train, w_train,
              cat_features, numeric_features, all_features, pos_weight=None):
        return self.dm.create_random_forest_model(
            x_train[numeric_features],
            y_train,
            self.cfg.rf_params,
            w_train
        )


class SVMTrainer(BaseTrainer):
    def train(self, x_train_scaled, x_train, y_train, w_train,
              cat_features, numeric_features, all_features, pos_weight=None):
        return self.dm.create_svm_model(
            x_train_scaled[numeric_features],
            y_train,
            self.cfg.svm_params
        )


class LGBMTrainer(BaseTrainer):
    def train(self, x_train_scaled, x_train, y_train, w_train,
              cat_features, numeric_features, all_features, pos_weight=None):
        return self.dm.create_lgbm_model(
            x_train[numeric_features],
            y_train,
            self.cfg.lgbm_params
        )


class CatBoostTrainer(BaseTrainer):
    def train(self, x_train_scaled, x_train, y_train, w_train,
              cat_features, numeric_features, all_features, pos_weight=None):
        return self.dm.create_catboost_model(
            x_train[all_features],   # includes STN
            y_train,
            self.cfg.catboost_params,
            cat_features
        )


# -------------------------------------------------------------------------
# Manager: runs all trainers
# -------------------------------------------------------------------------
class TrainingManager:
    def __init__(self, trainers: list[BaseTrainer]):
        self.trainers = trainers

    def train_all(self,
                  x_train_scaled, x_train, y_train, w_train,
                  cat_features, numeric_features, all_features,
                  pos_weight=None) -> Dict[str, Any]:
        models = {}
        for trainer in self.trainers:
            try:
                model = trainer.train(
                    x_train_scaled, x_train, y_train, w_train,
                    cat_features, numeric_features, all_features,
                    pos_weight
                )
                models[trainer.model_name] = model
            except Exception as e:
                # do not break if one model fails
                print(f"Training failed for {trainer.model_name}: {e}")
        return models
