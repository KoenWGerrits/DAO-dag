"""
Do not spread this Python script without the author's approval
Created on Fri Apr  5 10:56:01 2024
@author: Koen Gerrits
for more information contact: k.gerrits@vnog.nl or +31 6 14210001

"""
import pandas as pd
from xgboost import XGBClassifier, plot_importance
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import os
import re
import json
from catboost import CatBoostClassifier

class DevelopWeatherModel:
    """Class to develop meteorologic model"""
    def __init__(self):
        print("Initiating module")

    def load_data(self, data_path):
        """Loading data"""
        dataframe = pd.read_csv(data_path)
        return dataframe

    def hyperparameter_tuning(self, x_train, y_train, parameters, model_estimator):
        """Hyperparameter tuning to optimize model"""
        # Initiate grid search for optimal parameter combinations
        grid_search = GridSearchCV(
            estimator=model_estimator,
            param_grid=parameters,
            scoring = 'f1',
            n_jobs = 10,
            cv = 10,
            verbose=1)

        # Fit different models
        grid_search.fit(x_train, y_train)
        print(grid_search.best_estimator_)
        return grid_search.best_estimator_

    def create_xgb_model(self, x_train, y_train, parameters, weights):
        """Developing model based on parameters"""
        # Initiate XGB model
        xgb_classifier = XGBClassifier(objective='binary:logistic',
                                       enable_categorical=True,
                                       max_depth=parameters["max_depth"],
                                       learning_rate=parameters["learning_rate"],
                                       n_estimators=parameters["n_estimators"],
                                       scale_pos_weight=parameters["scale_pos_weight"],
                                       nthread=4)
        # Fit model on data
        try:
            xgb_classifier.fit(x_train, y_train, sample_weight=weights)
        except KeyError:
            xgb_classifier.fit(x_train, y_train)
        return xgb_classifier

    def create_logreg_model(self, x_train, y_train, parameters, weights):
        """Developing logistic regression model"""
        # Set model parameters
        logreg = LogisticRegression(class_weight="balanced",
                                    penalty = parameters["penalty"],
                                    C=parameters["C"],
                                    solver=parameters["solver"],
                                    max_iter=parameters["max_iter"])
        # Fit model on data
        model = logreg.fit(x_train, y_train, sample_weight=weights)
        return model

    def create_random_forest_model(self, x_train, y_train, parameters, weights):
        """Developing Random Forest Classifier model"""
        rf = RandomForestClassifier(
            n_estimators=parameters["n_estimators"],
            max_depth=parameters["max_depth"],
            min_samples_split=parameters["min_samples_split"],
            min_samples_leaf=parameters["min_samples_leaf"],
            class_weight=parameters.get("class_weight", "balanced"),
            random_state=parameters.get("random_state", 42)
        )
        model = rf.fit(x_train, y_train, sample_weight = weights)
        return model


    def create_svm_model(self, x_train, y_train, parameters):
        """Developing Support Vector Machine model"""
        svm = SVC(
            kernel=parameters["kernel"],
            C=parameters["C"],
            gamma=parameters["gamma"],
            class_weight=parameters.get("class_weight", "balanced"),
            probability=True
        )
        model = svm.fit(x_train, y_train)
        return model

    def create_lgbm_model(self, x_train, y_train, parameters):
        """Developing LightGBM Classifier model"""
        lgbm = lgb.LGBMClassifier(
            n_estimators=parameters["n_estimators"],
            learning_rate=parameters["learning_rate"],
            num_leaves=parameters["num_leaves"],
            max_depth=parameters["max_depth"],
            class_weight=parameters.get("class_weight", "balanced"),
            random_state=parameters.get("random_state", 42)
        )
        model = lgbm.fit(x_train, y_train)
        return model
    
    def create_catboost_model(self, x_train, y_train, parameters, cat_features):
        """Developing CatBoost Classifier model"""

        catboost = CatBoostClassifier(
            iterations=parameters["iterations"],
            learning_rate=parameters["learning_rate"],
            depth=parameters["max_depth"],
            l2_leaf_reg=parameters["l2_leaf_reg"],
            class_weights=parameters.get("class_weights", [1,1]),
            random_seed=parameters.get("random_seed", 42),
            verbose=False,
            eval_metric=parameters.get("eval_metric", "F1"),
            loss_function=parameters.get("loss_function", "Logloss")
        )

        model = catboost.fit(x_train, y_train, cat_features=cat_features)

        return model

    @staticmethod
    def save_tuning_results(results: dict, model_name: str, output_dir: str):
        """Save best parameters + cv results to hyperparameter_tuning folder"""
        tuning_dir = os.path.join(output_dir, "hyperparameter_tuning")
        os.makedirs(tuning_dir, exist_ok=True)

        # 1. Save best params + score
        best_path = os.path.join(tuning_dir, f"{model_name}_best.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(
                {"best_params": results["best_params"], "best_score": results["best_score"]},
                f, indent=2
            )

        # 2. Save cv_results to CSV
        cv_df = pd.DataFrame(results["cv_results"])
        cv_df.to_csv(os.path.join(tuning_dir, f"{model_name}_cv_results.csv"), index=False)

    def evaluate_model(
        self,
        model,
        x_test: pd.DataFrame,
        station_data: pd.DataFrame,
        y_test: pd.Series,
        model_type: str,
        output_dir: str | None = None,
        saved_name: str | None = None
    ):
        """Evaluate model with confusion matrix, probability plots, and save performance metrics"""
        print(f"Start evaluating model: {model_type}")

        # --- Predictions ---
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(x_test)[:, 1]
        else:
            # final fallback = hard predictions (not ideal)
            y_proba = model.predict(x_test)

        y_pred = (y_proba >= 0.5).astype(int)

        # --- Metrics ---
        f1 = metrics.f1_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, zero_division=0)
        recall = metrics.recall_score(y_test, y_pred, zero_division=0)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print(f"{model_type} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # --- Confusion Matrix ---
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        categories = ["Geen natuurbrand", "Wel natuurbrand"]
        fig, ax = plt.subplots()
        sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt="d",
                    xticklabels=categories, yticklabels=categories, ax=ax)
        ax.set_title(f'{model_type} Confusion Matrix', y=1.1)
        ax.set_ylabel('Werkelijk')
        ax.set_xlabel('Voorspeld')
        plt.tight_layout()
        plt.show(block=False)

        # --- Red/blue histogram of probabilities ---
        df_eval = pd.DataFrame({"prediction": y_proba, "fire": y_test.values})
        fire_pred = df_eval.loc[df_eval["fire"] == 1, "prediction"]
        no_fire_pred = df_eval.loc[df_eval["fire"] == 0, "prediction"]

        plt.figure(figsize=(10, 6))
        sns.histplot(fire_pred, color='r', label='Wel natuurbrand',
                    stat='count', kde=False, bins=20, alpha=0.6)
        sns.histplot(no_fire_pred, color='b', label='Geen natuurbrand',
                    stat='count', kde=False, bins=20, alpha=0.6)
        plt.xlabel('Voorspelde kans op natuurbrand')
        plt.ylabel("Aantal dagen")
        plt.title(f"Distributie voorspelde kansen ({model_type})")
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

        # --- Calibration-style barplot ---
        bins = np.linspace(0, 1, 21)
        df_eval['bin'] = pd.cut(df_eval['prediction'], bins=bins, include_lowest=True)
        bin_midpoints = [interval.mid for interval in df_eval['bin'].cat.categories]

        bin_summary = df_eval.groupby('bin', observed=False)['fire'].agg(['count', 'sum'])
        bin_summary['percent_fire'] = (bin_summary['sum'] / bin_summary['count']) * 100

        plt.figure(figsize=(10, 6))
        plt.bar(bin_midpoints, bin_summary['percent_fire'],
                width=0.05, align="center", color='firebrick', edgecolor='black')
        plt.xlabel('Voorspelde kans (bins)')
        plt.ylabel('Percentage dagen met natuurbrand')
        plt.title(f'Kalibratieplot: % natuurbrand per kans-bin ({model_type})')
        plt.xticks(np.linspace(0, 1, 11))
        plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show(block=True)

        # --- Save metrics ---
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            perf_file = os.path.join(output_dir, "model_performance.csv")

            params = model.get_params() if hasattr(model, "get_params") else {}
            perf_dict = {
                "model_name": saved_name if saved_name else model_type,
                "params": json.dumps(params),
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy
            }

            if os.path.exists(perf_file):
                df_perf = pd.read_csv(perf_file)
                df_perf = pd.concat([df_perf, pd.DataFrame([perf_dict])], ignore_index=True)
            else:
                df_perf = pd.DataFrame([perf_dict])

            df_perf.to_csv(perf_file, index=False)
            print(f"Saved performance metrics to {perf_file}")

            # --- Per-station evaluation ---
            stations = station_data["STN"].unique()
            per_station_rows = []
            station_data = station_data.reset_index(drop=True)
            # Ensure alignment
            station_data["target"] = y_test.values  # ensure alignment

            for stn in stations:
                stn_data = station_data[station_data["STN"] == stn].copy()
                stn_y = stn_data["target"]
                stn_pred = y_proba[stn_data.index] if hasattr(model, "predict_proba") else model.predict(x_test.loc[stn_data.index])
                stn_pred_labels = (stn_pred >= 0.5).astype(int)

                # Compute metrics
                stn_acc = metrics.accuracy_score(stn_y, stn_pred_labels)
                stn_prec = metrics.precision_score(stn_y, stn_pred_labels, zero_division=0)
                stn_rec = metrics.recall_score(stn_y, stn_pred_labels, zero_division=0)
                stn_f1 = metrics.f1_score(stn_y, stn_pred_labels, zero_division=0)

                per_station_rows.append({
                    "model_name": model_type,
                    "station": stn,
                    "accuracy": stn_acc,
                    "precision": stn_prec,
                    "recall": stn_rec,
                    "f1": stn_f1
                })

                print(f"[Station {stn}] Accuracy: {stn_acc:.4f}, F1: {stn_f1:.4f}, Precision: {stn_prec:.4f}, Recall: {stn_rec:.4f}")



    def plot_logreg(self, model, x_train_scaled, columns):

        # Store mean values of variables
        mean_values = x_train_scaled.mean()
        # Generate ranges for each variable
        ranges = {}
        for col in columns:
            ranges[col] = np.linspace(x_train_scaled[col].min(), x_train_scaled[col].max(), 100)
        # Store variable names
        variables = x_train_scaled.columns.tolist()

        # Iterate through variables to generate plots
        for var in variables:
            # Create DataFrame with variable range
            df = pd.DataFrame(ranges[var], columns=[var])

            # Fill other variables with mean values
            for col in variables:
                if col != var:
                    df[col] = mean_values[col]


            # Predict using logistic regression model
            df['prediction'] = model.predict_proba(df[variables])[:,1]

            # Plot
            plt.figure()
            plt.plot(df[var], df['prediction'])
            plt.xlabel('Variable Values')
            plt.ylabel('Model Output')
            plt.title(f'Effect of {var} on Model Output')
            plt.grid(True)
            plt.ylim(0, 1)
            plt.show()


    def find_latest_model(self, folder_path, model_type):
        files = []
        version_pattern = re.compile(r"([0-9]+(?:\.[0-9]+)?)")

        for file_name in os.listdir(folder_path):
            if model_type in file_name:
                match = version_pattern.search(file_name)
                if match:
                    version = float(match.group(0))
                    files.append((version, file_name))

        if not files:
            print("No files with version found.")
            return None, 1.0  # Start from version 1.0 if none found

        # Sort by version number descending and return the highest version
        latest_version, latest_file = max(files, key=lambda x: x[0])
        return latest_file, latest_version

    def generate_new_model_name(self, latest_version, model_type):
        new_version = round(latest_version + 0.1, 1)
        return f"{model_type}_v_{new_version:.1f}.joblib"
