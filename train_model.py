
import datetime
import json
import sys
import time
from collections import Counter

import joblib
import matplotlib
matplotlib.use("Agg")                       # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import churn_utils as U

# TUTORIAL NOTE: the tutorial calls train_test_split WITHOUT random_state, so every run
# gives a different split and different metrics. A seed is added so your results
# are reproducible. Set SPLIT_SEED = None to match the tutorial exactly.
SPLIT_SEED = 42
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def evaluate(model, X_test, y_test):
    """The tutorial's evaluation prints (recall, precision, accuracy, F1, report,
    confusion matrix, ROC-AUC, ROC curve) collected into one dictionary."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    keep = np.linspace(0, len(fpr) - 1, min(300, len(fpr))).astype(int)  # keep the JSON small
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "roc_curve": {"fpr": fpr[keep].tolist(), "tpr": tpr[keep].tolist()},
    }, y_proba


# ============================================================ 1. load + clean (tutorial)
try:
    data = U.load_data()
except (FileNotFoundError, ValueError) as err:
    sys.exit(f"ERROR: {err}")
log(f"Loaded data: {data.shape}")
data, n_dropped = U.clean_data(data)
log(f"Total dropped rows with NaNs: {n_dropped}")

# ============================================================ 2. features + target (tutorial)
numeric_features, categorical_features = U.NUMERIC_FEATURES, U.CATEGORICAL_FEATURES
X = data[numeric_features + categorical_features]
y = data[U.TARGET]
log(f"Class counts: {dict(Counter(y))}  (imbalanced, so class_weight / scale_pos_weight are used)")

# ============================================================ 3. stratified split (tutorial)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SPLIT_SEED)
y_train = y_train.map({"Yes": 1, "No": 0})
y_test = y_test.map({"Yes": 1, "No": 0})
y = y.map({"Yes": 1, "No": 0})
log(f"Churn rate: all={y.mean():.3f} train={y_train.mean():.3f} test={y_test.mean():.3f}")

# ============================================================ 4. preprocessing (tutorial)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

# ============================================================ 5. Logistic Regression (tutorial)
clf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(class_weight="balanced", random_state=42)),
])
clf_pipeline.fit(X_train, y_train)
lr_result, _ = evaluate(clf_pipeline, X_test, y_test)
log(f"Logistic Regression  AUC={lr_result['roc_auc']:.3f} recall={lr_result['recall']:.3f}")

# ============================================================ 6. Random Forest (default params)
# TUTORIAL NOTE (ASSUMPTION): the tutorial code uses `rf_model` but never defines it.
# The grid search uses the prefix "model__", so it must be a Pipeline whose
# classifier step is named "model". The Random Forest parameters are NOT in the code
# you supplied. class_weight="balanced" is assumed because the tutorial handles
# imbalance the same way for Logistic Regression. If your video shows different
# parameters, change ONLY the RandomForestClassifier(...) line below.
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(class_weight="balanced", random_state=42)),
])
rf_model.fit(X_train, y_train)
rf_default_result, _ = evaluate(rf_model, X_test, y_test)
log(f"Random Forest (default)  AUC={rf_default_result['roc_auc']:.3f}")

# ============================================================ 7. cross-validation (tutorial)
# cv=5 on the TRAINING data only (not the test set)
cv_lr = cross_val_score(clf_pipeline, X_train, y_train, scoring="roc_auc", cv=5)
cv_rf = cross_val_score(rf_model, X_train, y_train, scoring="roc_auc", cv=5)
log(f"CV AUC: LR={cv_lr.mean():.3f}  RF={cv_rf.mean():.3f}")

# ============================================================ 8. hyperparameter tuning (tutorial)
param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf": [1, 5],
}
grid_rf = GridSearchCV(rf_model, param_grid, scoring="roc_auc", cv=3, n_jobs=-1, verbose=2)
log("Running GridSearchCV (24 combinations x 3 folds = 72 fits). This takes a few minutes...")
grid_rf.fit(X_train, y_train)
log(f"Best RF params: {grid_rf.best_params_}")
log(f"Best RF CV AUC: {grid_rf.best_score_:.4f}")
best_rf = grid_rf.best_estimator_
best_idx = grid_rf.best_index_
rf_tuned_result, y_proba_rf = evaluate(best_rf, X_test, y_test)
log(f"Random Forest (tuned)  AUC={rf_tuned_result['roc_auc']:.3f} recall={rf_tuned_result['recall']:.3f}")

# ============================================================ 9. feature importance (tutorial)
feature_names = best_rf.named_steps["preprocessor"].get_feature_names_out()
importances = best_rf.named_steps["model"].feature_importances_
feat_imp = (pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values(by="importance", ascending=False))
top_features = feat_imp.head(15)

# ============================================================ 10. threshold analysis (tutorial)
threshold_rows = []
for t in np.arange(0.1, 0.9, 0.05):
    y_pred_custom = (y_proba_rf >= t).astype(int)
    # zero_division=0 only hides a warning when no customer is flagged at a high threshold
    threshold_rows.append({
        "threshold": round(float(t), 2),
        "precision": precision_score(y_test, y_pred_custom, zero_division=0),
        "recall": recall_score(y_test, y_pred_custom),
        "f1": f1_score(y_test, y_pred_custom),
    })

# ============================================================ 11. risk levels (tutorial)
risk_level = pd.cut(pd.Series(y_proba_rf), bins=[0, 0.3, 0.6, 1],
                    labels=["Low Risk", "Medium Risk", "High Risk"])
risk_counts = risk_level.value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"]).fillna(0)

# ============================================================ 12. save models (tutorial)
U.MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(clf_pipeline, U.LR_MODEL_PATH)
joblib.dump(best_rf, U.RF_MODEL_PATH)
log("Saved logistic_model.pkl and rf_model.pkl")

# ============================================================ 13. SHAP (tutorial)
shap_ok = False
try:
    import shap
    X_train_transformed = best_rf.named_steps["preprocessor"].transform(X_train)
    # TUTORIAL NOTE: SHAP needs a dense array, so a sparse matrix is converted first
    if sparse.issparse(X_train_transformed):
        X_train_transformed = X_train_transformed.toarray()
    explainer = shap.TreeExplainer(best_rf.named_steps["model"])
    log("Computing SHAP values (can take a few minutes)...")
    shap_values = explainer.shap_values(X_train_transformed)
    # TUTORIAL NOTE: newer SHAP returns one 3D array (rows, features, classes);
    # older SHAP returns a list per class. Either way we take class 1 = churn.
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]
    for plot_type, path in [(None, U.SHAP_SUMMARY_PATH), ("bar", U.SHAP_BAR_PATH)]:
        plt.figure(figsize=(9, 6.5))
        kwargs = {"plot_type": plot_type} if plot_type else {}
        shap.summary_plot(sv, X_train_transformed, feature_names=feature_names, show=False, **kwargs)
        plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
        plt.close()
    shap_ok = True
    log("Saved SHAP plots")
except Exception as err:                       # training must not fail because of SHAP
    log(f"SHAP step skipped: {err}")

# ============================================================ 14. XGBoost (tutorial)
neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
xgb_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric="logloss")),
])
xgb_model.fit(X_train, y_train)
xgb_result, _ = evaluate(xgb_model, X_test, y_test)
joblib.dump(xgb_model, U.XGB_MODEL_PATH)
log(f"XGBoost  AUC={xgb_result['roc_auc']:.3f} recall={xgb_result['recall']:.3f}")

# ============================================================ 15. save every REAL number (replaces the tutorial's hardcoded performance_df)
cv_info = {"scope": "5-fold cross-validation on training data"}
models = {
    "Logistic Regression": {**lr_result, "cv": {"mean": cv_lr.mean(), "std": cv_lr.std(), "folds": 5}},
    "Random Forest (default)": {**rf_default_result, "cv": {"mean": cv_rf.mean(), "std": cv_rf.std(), "folds": 5}},
    "Random Forest (tuned)": {**rf_tuned_result, "cv": {
        "mean": grid_rf.best_score_,
        "std": grid_rf.cv_results_["std_test_score"][best_idx], "folds": 3}},
    "XGBoost": {**xgb_result, "cv": None},    # the tutorial does not cross-validate XGBoost
}
meta = {
    "final_model": "Random Forest (tuned)",
    "total_customers": len(data), "churned_customers": int(y.sum()), "churn_rate": float(y.mean()),
    "dropped_rows": n_dropped, "n_train": len(X_train), "n_test": len(X_test), "test_size": 0.3,
    "split_seed": SPLIT_SEED, "features": U.FEATURES,
    "grid": param_grid, "best_params": grid_rf.best_params_, "grid_fits": 24 * 3,
    "xgb_scale_pos_weight": float(scale_pos_weight),
    "shap_available": shap_ok,
    "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
}
output = {
    "meta": meta, "models": models,
    "feature_importance": top_features.to_dict("records"),
    "threshold_analysis": threshold_rows,
    "risk_levels": {k: int(v) for k, v in risk_counts.items()},
}
with open(U.METRICS_PATH, "w") as f:
    json.dump(output, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
log("Saved metrics.json. Done.")
