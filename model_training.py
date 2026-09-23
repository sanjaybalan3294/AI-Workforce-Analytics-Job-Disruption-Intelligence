"""
model_training.py
=============================================================================
Tier 3: Predictive Modeling & Leakage Prevention Pipeline
Trains baseline and challenger models, enforces target leakage protection,
evaluates performance metrics, and computes commercial trade-offs.
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


class ModelTrainingPipeline:
    """
    Manages data preprocessing, leakage-proof feature selection,
    model training, benchmarking, and commercial risk evaluation.
    """

    def __init__(self, data_path: str = "cleaned_ai_job_trends.csv"):
        self.data_path = data_path
        self.df = None
        self.artifacts = {}

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Cleaned dataset not found at {self.data_path}. Run data_pipeline.py first.")
        self.df = pd.read_csv(self.data_path)
        print(f"[INFO] Loaded {len(self.df)} records for predictive modeling.")

    def prepare_data(self, target_col: str = "Decline_Risk_Flag"):
        """
        Enforces strict leakage protection:
        - Drops direct target derivatives and future-dated columns.
        - Drops high-cardinality raw text identifiers (Job Title).
        """
        leakage_cols = [
            "Job Status",
            "Decline_Risk_Flag",
            "High_Automation_Risk_Flag",
            "Projected Openings (2030)",   # Future lookahead bias
            "Net_Projected_Growth",        # Derived from 2030 future
            "Growth_Rate_Pct",             # Derived from 2030 future
            "Job Title"                    # 639 raw IDs causing memorization
        ]

        # Target variable extraction
        y = self.df[target_col].copy()

        # Features
        feature_cols = [c for c in self.df.columns if c not in leakage_cols]
        X = self.df[feature_cols].copy()

        num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_features = X.select_dtypes(include=["object"]).columns.tolist()

        print(f"[INFO] Target: {target_col}")
        print(f"[INFO] Protected Predictor Features ({len(feature_cols)} total):")
        print(f"  - Numeric ({len(num_features)}): {num_features}")
        print(f"  - Categorical ({len(cat_features)}): {cat_features}")

        # Train/Test Split (80/20 Stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        return X_train, X_test, y_train, y_test, num_features, cat_features

    def build_preprocessor(self, num_features: list, cat_features: list) -> ColumnTransformer:
        """Constructs scikit-learn ColumnTransformer for scaling and one-hot encoding."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features)
            ]
        )
        return preprocessor

    def train_and_evaluate(self):
        """Trains Logistic Regression and Random Forest models and evaluates metrics."""
        self.load_data()
        X_train, X_test, y_train, y_test, num_features, cat_features = self.prepare_data()

        preprocessor = self.build_preprocessor(num_features, cat_features)

        # 1. Baseline Model: L2 Regularized Logistic Regression
        lr_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
        ])

        # 2. Challenger Model: Random Forest Classifier
        rf_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=15,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            ))
        ])

        models = {
            "Logistic Regression (Baseline)": lr_pipeline,
            "Random Forest (Challenger)": rf_pipeline
        }

        results = {}

        for model_name, pipeline in models.items():
            print(f"\n[INFO] Training {model_name}...")
            pipeline.fit(X_train, y_train)

            # Predictions and Probabilities
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            y_prob_train = pipeline.predict_proba(X_train)[:, 1]
            y_prob_test = pipeline.predict_proba(X_test)[:, 1]

            # Metric Evaluation
            acc_test = accuracy_score(y_test, y_pred_test)
            prec_test = precision_score(y_test, y_pred_test, zero_division=0)
            rec_test = recall_score(y_test, y_pred_test)
            f1_test = f1_score(y_test, y_pred_test)
            roc_auc_test = roc_auc_score(y_test, y_prob_test)
            pr_auc_test = average_precision_score(y_test, y_prob_test)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred_test)
            tn, fp, fn, tp = cm.ravel()

            # ROC & PR Curves
            fpr, tpr, roc_thresh = roc_curve(y_test, y_prob_test)
            precisions, recalls, pr_thresh = precision_recall_curve(y_test, y_prob_test)

            # Commercial Loss Optimization Curve
            # Business assumptions:
            # Cost of False Positive (wasted reskilling): $3,500
            # Cost of False Negative (unmitigated displacement / emergency severance): $18,500
            cost_fp = 3500
            cost_fn = 18500
            thresholds = np.linspace(0.1, 0.9, 81)
            commercial_costs = []
            for t in thresholds:
                pred_t = (y_prob_test >= t).astype(int)
                cm_t = confusion_matrix(y_test, pred_t)
                _, fp_t, fn_t, _ = cm_t.ravel()
                total_cost = (fp_t * cost_fp) + (fn_t * cost_fn)
                commercial_costs.append({
                    "threshold": t,
                    "fp_count": int(fp_t),
                    "fn_count": int(fn_t),
                    "total_cost_usd": float(total_cost)
                })

            opt_cost_entry = min(commercial_costs, key=lambda x: x["total_cost_usd"])

            results[model_name] = {
                "pipeline": pipeline,
                "metrics": {
                    "accuracy": float(acc_test),
                    "precision": float(prec_test),
                    "recall": float(rec_test),
                    "f1_score": float(f1_test),
                    "roc_auc": float(roc_auc_test),
                    "pr_auc": float(pr_auc_test)
                },
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "matrix": cm.tolist()
                },
                "roc_curve": {
                    "fpr": fpr.tolist()[::max(1, len(fpr)//100)],
                    "tpr": tpr.tolist()[::max(1, len(tpr)//100)]
                },
                "pr_curve": {
                    "precision": precisions.tolist()[::max(1, len(precisions)//100)],
                    "recall": recalls.tolist()[::max(1, len(recalls)//100)]
                },
                "commercial_tradeoff": {
                    "cost_fp": cost_fp,
                    "cost_fn": cost_fn,
                    "cost_curve": commercial_costs,
                    "optimal_threshold": opt_cost_entry["threshold"],
                    "min_cost_usd": opt_cost_entry["total_cost_usd"]
                }
            }

            print(f"  - Accuracy:  {acc_test:.4f}")
            print(f"  - Precision: {prec_test:.4f}")
            print(f"  - Recall:    {rec_test:.4f}")
            print(f"  - ROC-AUC:   {roc_auc_test:.4f}")
            print(f"  - Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            print(f"  - Economically Optimal Threshold: {opt_cost_entry['threshold']:.2f} (Total Loss: ${opt_cost_entry['total_cost_usd']:,.0f})")

        # Feature Importances / Coefficients
        ohe = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        encoded_cat_names = ohe.get_feature_names_out(cat_features).tolist()
        all_feature_names = num_features + encoded_cat_names

        rf_importances = rf_pipeline.named_steps["classifier"].feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": all_feature_names,
            "importance": rf_importances
        }).sort_values("importance", ascending=False)

        # Logistic Regression Coeffs
        lr_coeffs = lr_pipeline.named_steps["classifier"].coef_[0]
        lr_coeffs_df = pd.DataFrame({
            "feature": all_feature_names,
            "coefficient": lr_coeffs
        }).sort_values("coefficient", key=abs, ascending=False)

        # Store Artifacts
        self.artifacts = {
            "results": results,
            "feature_importance_rf": feature_importance_df.to_dict(orient="records"),
            "coefficients_lr": lr_coeffs_df.to_dict(orient="records"),
            "num_features": num_features,
            "cat_features": cat_features,
            "test_sample": X_test.head(10).to_dict(orient="records"),
            "test_sample_labels": y_test.head(10).tolist(),
            "target_col": "Decline_Risk_Flag"
        }

        # Save to disk
        output_file = "model_artifacts.pkl"
        joblib.dump(self.artifacts, output_file)
        print(f"\n[SUCCESS] Model artifacts and evaluation metrics saved to {output_file}!")
        return self.artifacts


if __name__ == "__main__":
    trainer = ModelTrainingPipeline()
    trainer.train_and_evaluate()
