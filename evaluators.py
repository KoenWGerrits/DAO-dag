"""
evaluators.py

Defines BaseEvaluator and concrete evaluators for different ML algorithms.
Each evaluator wraps the corresponding evaluate_model method from
DevelopWeatherModel, selecting the right feature set.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd


# -------------------------------------------------------------------------
# Abstract Base Evaluator
# -------------------------------------------------------------------------
class BaseEvaluator(ABC):
    def __init__(self, cfg: Any, dm: Any, model_name: str):
        """
        cfg: Config object
        dm: DevelopWeatherModel instance (with evaluate_model method)
        model_name: short identifier (e.g., "XGB", "logreg")
        """
        self.cfg = cfg
        self.dm = dm
        self.model_name = model_name

    @abstractmethod
    def get_eval_features(self,
                          x_test_scaled: pd.DataFrame,
                          x_test: pd.DataFrame,
                          all_features: list,
                          numeric_features: list,
                          cat_features: list) -> pd.DataFrame:
        """Return the correct feature set for evaluation"""
        pass

    def evaluate(self,
                 model: Any,
                 x_test_scaled: pd.DataFrame,
                 x_test: pd.DataFrame,
                 y_test: pd.Series,
                 station_data: pd.DataFrame,
                 all_features: list,
                 numeric_features: list,
                 cat_features: list,
                 output_dir: str,
                 saved_name: str):
        """
        Call dm.evaluate_model with the right features.
        """
        X_eval = self.get_eval_features(x_test_scaled,
                                        x_test,
                                        all_features,
                                        numeric_features,
                                        cat_features)

        # Call the shared evaluation method from develop_model.py
        self.dm.evaluate_model(
            model,
            X_eval,
            station_data,
            y_test,
            self.model_name,
            output_dir=output_dir,
            saved_name=saved_name  # not used here
        )


# -------------------------------------------------------------------------
# Concrete Evaluators
# -------------------------------------------------------------------------
class XGBEvaluator(BaseEvaluator):
    def get_eval_features(self, x_test_scaled, x_test, all_features, numeric_features, cat_features):
        return x_test[numeric_features]


class LogRegEvaluator(BaseEvaluator):
    def get_eval_features(self, x_test_scaled, x_test, all_features, numeric_features, cat_features):
        return x_test_scaled[numeric_features]


class RFEvaluator(BaseEvaluator):
    def get_eval_features(self, x_test_scaled, x_test, all_features, numeric_features, cat_features):
        return x_test[numeric_features]


class SVMEvaluator(BaseEvaluator):
    def get_eval_features(self, x_test_scaled, x_test, all_features, numeric_features, cat_features):
        return x_test_scaled[numeric_features]


class LGBMEvaluator(BaseEvaluator):
    def get_eval_features(self, x_test_scaled, x_test, all_features, numeric_features, cat_features):
        return x_test[numeric_features]


class CatBoostEvaluator(BaseEvaluator):
    def get_eval_features(self, x_test_scaled, x_test, all_features, numeric_features, cat_features):
        # CatBoost gets categorical + numeric (incl. STN)
        return x_test[all_features]


# -------------------------------------------------------------------------
# Manager: run evaluation for all models
# -------------------------------------------------------------------------
class EvaluationManager:
    def __init__(self, evaluators: list[BaseEvaluator], saved_names: Dict[str, str] = {}):
        self.evaluators = evaluators
        self.saved_names = saved_names

    def evaluate_all(self,
                     models: Dict[str, Any],
                     x_test_scaled: pd.DataFrame,
                     x_test: pd.DataFrame,
                     y_test: pd.Series,
                     station_data: pd.DataFrame,
                     all_features: list,
                     numeric_features: list,
                     cat_features: list,
                     output_dir: str):
        for evaluator in self.evaluators:
            if evaluator.model_name not in models:
                continue  # skip if model wasn't trained
            model = models[evaluator.model_name]
            saved_name = self.saved_names.get(evaluator.model_name)
            try:
                evaluator.evaluate(
                                    model=model,
                                    x_test_scaled=x_test_scaled,
                                    x_test=x_test,
                                    y_test=y_test,
                                    station_data=station_data,
                                    all_features=all_features,
                                    numeric_features=numeric_features,
                                    cat_features=cat_features,
                                    output_dir=output_dir,
                                    saved_name=saved_name
                )
            except Exception as e:
                print(f"Evaluation failed for {evaluator.model_name}: {e}")
