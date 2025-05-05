import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform, loguniform
from joblib import dump

# -------------------------------------------------
# Functions
# -------------------------------------------------

def compute_metrics_from_cm(cm: np.ndarray):
    """
    Compute precision, recall, F1, and accuracy from a confusion matrix.
    cm[i, j] = number of samples with true class i predicted as class j.
    Returns a dict of metrics.
    """
    tn = cm[0, 0]
    fn = cm[1, 0]
    fp = cm[0, 1]
    tp = cm[1, 1]
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tn + fp + fn + tp + 1e-10)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "TN": float(tn),
        "FN": float(fn),
        "FP": float(fp),
        "TP": float(tp)
    }
    return metrics

# --------------------------------------------------------------
# Main script
# --------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)

    # 1) Load data
    file_path = os.path.join(script_dir, "early_vs_others.tsv")
    df = pd.read_csv(file_path, sep="\t")

    feature_cols = [col for col in df.columns if col not in ['BioProject', 'stage', 'mediumBAL', 'mediumNPS', 'mediumSputum', 'mediumTS', 'adult_pediatric', 'continent', 'sex']]
    groups = df['BioProject']
    y = df['stage']
    meta = df[['mediumBAL', 'mediumNPS', 'mediumSputum', 'mediumTS', 'adult_pediatric', 'continent', 'sex']]

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')

    # Total Sum Scaling
    df[feature_cols] = df[feature_cols].div(df[feature_cols].sum(axis=1), axis=0)

    # Log-normalization with offset
    df[feature_cols] = np.log(df[feature_cols] + 1)

    # Zero-centering batch correction
    for b in df["BioProject"].unique():
        idx = (df["BioProject"] == b)
        batch_features = df.loc[idx, feature_cols]
        batch_means = batch_features.mean(axis=0)
        df.loc[idx, feature_cols] = batch_features - batch_means

    X = pd.concat([meta, df[feature_cols]], axis=1)

    # 2) Define cross-validator + models
    logo = LeaveOneGroupOut()
    models = {
        "Random Forest": {
            "model": RandomForestClassifier(),
            "params": {
                "classifier__n_estimators": randint(100, 1000),
                "classifier__max_depth": randint(5, 20),
                "classifier__min_samples_split": randint(2, 20),
                "classifier__min_samples_leaf": randint(1, 20),
                "classifier__max_features": uniform(0.1, 0.9)
            },
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier(),
            "params": {
                "classifier__n_neighbors": randint(2, 50),
                "classifier__weights": ["uniform", "distance"],
                "classifier__p": [1, 2]
            },
        },
        "SVM (RBF)": {
            "model": SVC(kernel='rbf', probability=True),
            "params": {
                "classifier__C": loguniform(1e-2, 1e+2),
                "classifier__gamma": loguniform(1e-2, 1e+2),
                "classifier__shrinking": [True, False]
            },
        },
        "Logistic Regression-L2": {
            "model": LogisticRegression(max_iter=10000),
            "params": {
                "classifier__solver": ["sag"],
                "classifier__C": loguniform(1e-4, 1e+2),
                "classifier__penalty": ["l2"],
                "classifier__fit_intercept": [True],
            },
        },
        "SVM (Linear)": {
            "model": SVC(kernel='linear', probability=True),
            "params": {
                "classifier__C": loguniform(1e-2, 1e+2),
                "classifier__shrinking": [True, False]
            },
        },
        "XGBoost": {
            "model": XGBClassifier(
                eval_metric="logloss",
                objective="binary:logistic",
                n_jobs=4
            ),
            "params": {
                "classifier__n_estimators": randint(100, 1000),
                "classifier__max_depth": randint(5, 15),
                "classifier__learning_rate": loguniform(1e-4, 0.1),
                "classifier__subsample": uniform(0.5, 0.5),
                "classifier__colsample_bytree": uniform(0.5, 0.5),
                "classifier__gamma": loguniform(1e-5, 1e+1),
                "classifier__reg_alpha": loguniform(1e-3, 1e+1),
                "classifier__reg_lambda": loguniform(1e-3, 1e+1)
            }
        },
        "Elastic Net": {
            "model": LogisticRegression(solver="saga", penalty="elasticnet", max_iter=10000),
            "params": {
                "classifier__C": loguniform(1e-1, 2e+1),
                "classifier__l1_ratio": uniform(0, 1),
                "classifier__fit_intercept": [True],
            },
        },
        "Logistic Regression-L1": {
            "model": LogisticRegression(solver="saga", max_iter=10000),
            "params": {
                "classifier__C": loguniform(1e-1, 2e+1),
                "classifier__penalty": ["l1"],
                "classifier__fit_intercept": [True],
            },
        }
    }

    results = {model_name: {} for model_name in models}
    scoring_metrics = {
        "accuracy": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score, zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0)
    }
    labels = [0, 1]
 

    # 3) Train each model in a pipeline
    for model_name, model_info in models.items():
        print(f"\n{'='*70}")
        print(f"Training and tuning model: {model_name}")
        print(f"{'='*70}\n")

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('smote', SMOTE(k_neighbors=10)),
            ('classifier', model_info["model"])
        ])
    
        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=model_info["params"],
            n_iter=500,
            cv=logo.split(X, y, groups),
            scoring=scoring_metrics,
            refit="accuracy",
            random_state=42,
            n_jobs=-1
        )
    
        random_search.fit(X, y)
        best_index = random_search.best_index_

        results[model_name] = {
            "best_model": random_search.best_estimator_,
            "best_params": random_search.best_params_,
            "f1": random_search.cv_results_['mean_test_f1'][best_index],
            "accuracy": random_search.cv_results_['mean_test_accuracy'][best_index],
            "precision": random_search.cv_results_['mean_test_precision'][best_index],
            "recall": random_search.cv_results_['mean_test_recall'][best_index],
            "fold_metrics": [],
            "average_metrics": {}
        }

        print("Best Score (accuracy):", random_search.best_score_)

        # Build a pipeline with the best hyperparameters
        best_params = random_search.best_params_
        best_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="most_frequent")),
            ('smote', SMOTE(k_neighbors=10)),
            ('classifier', model_info["model"])
        ])
        best_pipeline.set_params(**best_params)

        # 4) Evaluate with default threshold of 0.5
        t_default = 0.5
        overall_cm = np.zeros((2, 2))

        feature_importance_per_fold = []

        for fold_index, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
            best_pipeline.fit(X_train, y_train)
            fold_size =len(test_idx)

            # extract selected features 
            selected_features = X.columns[selected_mask].tolist()

            classifier = best_pipeline.named_steps['classifier']
            if hasattr(classifier, 'feature_importances_'):
                importance = classifier.feature_importances_
            elif hasattr(classifier, 'coef_'):
                importance = np.abs(classifier.coef_[0])
            else:
                importance = None
            
            if importance is not None:
                fold_importance = dict(zip(selected_features, importance))
            else:
                fold_importance = {}

            feature_importance_per_fold.append({
                'fold':fold_size,
                'selected_features': selected_features,
                'importance_scores': fold_importance
            })

            y_prob = best_pipeline.predict_proba(X_test)[:, 1]
            y_pred_fold = (y_prob >= t_default).astype(int)

            cm_fold = confusion_matrix(y_test, y_pred_fold, labels=labels)
            metric = compute_metrics_from_cm(cm_fold)
            metric["test_fold_size"] = len(test_idx)
            metric["fold"] = fold_index
            results[model_name]["fold_metrics"].append(metric)

            normalized_cm_fold = cm_fold / len(test_idx)
            overall_cm += normalized_cm_fold

        overall_cm /= logo.get_n_splits(groups=groups)
        avg_metrics = compute_metrics_from_cm(overall_cm)
        results[model_name]["average_metrics"] = avg_metrics
        results[model_name]['feature_importance_per_fold'] = feature_importance_per_fold
    # Save the results
    # Fold-level metrics
    rows = []
    for model_name, model_dict in results.items():
        for fold_metric in model_dict["fold_metrics"]:
            rows.append({
                "model": model_name,
                "fold": fold_metric["fold"],
                "test_fold_size": fold_metric["test_fold_size"],
                "TP": fold_metric["TP"],
                "TN": fold_metric["TN"],
                "FP": fold_metric["FP"],
                "FN": fold_metric["FN"],
                "Accuracy": fold_metric["accuracy"]
            })
    df_results = pd.DataFrame(rows)
    df_results.to_csv("model_metrics_from_per_fold.csv", index=False)

    # Average metrics
    avg_rows = []
    for model_name, model_dict in results.items():
        avg_metric = model_dict["average_metrics"]
        avg_rows.append({
            "model": model_name,
            "Precision": avg_metric["precision"],
            "Recall": avg_metric["recall"],
            "F1": avg_metric["f1"],
            "Accuracy": avg_metric["accuracy"]
        })
    df_avg = pd.DataFrame(avg_rows)
    df_avg.to_csv("average_model_metrics.csv", index=False)

    # Best parameters
    best_params_dict = {model_name: model_info["best_params"] for model_name, model_info in results.items()}
    dump(best_params_dict, "best_params.joblib")
