# ChurnSense: Customer Churn Prediction System

A Streamlit web app that predicts whether a telecom customer is likely to churn.The app has two pages: **Predict Churn** and **Model Performance**.
<p align="center">
  <a href="https://churnsense-mlprediction.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Open_App-brightgreen?style=for-the-badge" alt="Live Demo">
  </a>
</p>


## Project structure
```
customer-churn-prediction/
├── app.py            # Streamlit app (2 pages)
├── train_model.py    
├── churn_utils.py    # shared paths, feature lists, data cleaning
├── requirements.txt
├── README.md
├── data/telecomechurn.csv
└── models/           # created by train_model.py
    ├── rf_model.pkl (final tuned Random Forest, used by the app)
    ├── logistic_model.pkl, xgb_model.pkl
    ├── metrics.json, shap_summary.png, shap_bar.png
```

## Setup and run (Windows PowerShell)
```powershell
cd customer-churn-prediction
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py      # about 2-5 minutes (GridSearchCV + SHAP)
streamlit run app.py       # opens http://localhost:8501
```

## Workflow (same order as the tutorial)
1. Load the CSV; convert `TotalCharges` to numbers and drop the 11 rows where it is blank.
2. Features: 3 numeric (`MonthlyCharges`, `TotalCharges`, `tenure`) and 16 categorical; target `Churn`.
3. Stratified split: 70% train / 30% test.
4. Preprocessing: median impute + standard scaling for numbers; most-frequent impute + one-hot encoding for categories.
5. Models: Logistic Regression (balanced), Random Forest (default), Random Forest tuned with GridSearchCV (3-fold, ROC-AUC, 72 fits), XGBoost (`scale_pos_weight`).
6. Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report, ROC curves, 5-fold cross-validation (LR and default RF), feature importance, threshold analysis, risk levels, SHAP.
7. The tuned Random Forest is the tutorial's "final model" and is the one used for predictions.

## Results (test set, 2,110 customers; cross-validation uses training data only)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.737 | 0.504 | 0.795 | 0.617 | 0.838 | 0.848 (5-fold) |
| Random Forest (default) | 0.781 | 0.618 | 0.456 | 0.525 | 0.813 | 0.823 (5-fold) |
| Random Forest (tuned) | 0.765 | 0.542 | 0.752 | 0.630 | 0.836 | 0.846 (3-fold) |
| XGBoost | 0.745 | 0.514 | 0.783 | 0.620 | 0.837 | not run |

Best tuned parameters: `{'model__max_depth': 10, 'model__min_samples_leaf': 5, 'model__min_samples_split': 2, 'model__n_estimators': 200}`.

## Things to know
- **Assumed line:** the tutorial code never defines `rf_model`. `train_model.py` uses `RandomForestClassifier(class_weight="balanced", random_state=42)`. Change only that line if your video differs.
- **Seed added:** the tutorial's `train_test_split` has no `random_state`, so `SPLIT_SEED = 42` was added for reproducible results (set it to `None` to match the tutorial exactly).
- **Compatibility fixes in the SHAP step:** sparse-to-dense conversion and support for both SHAP output formats.
- Predictions are probabilities, not guarantees. The data is a single snapshot from one company.
