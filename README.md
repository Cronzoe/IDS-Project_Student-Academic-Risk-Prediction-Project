# IDS-Project_Student-Academic-Risk-Prediction-Project
# Early Academic Risk Prediction System

An interactive machine learning web application that identifies students at risk of poor academic performance (`G3 < 10`) early—without relying on prior examination scores (`G1`, `G2`).

## Dataset & Preprocessing
- **Source:** UCI Machine Learning Repository (Student Performance — Mathematics & Portuguese courses)
- **Consolidation:** Merged 1,044 raw records and deduplicated across the 13 official student-identity attributes, yielding **662 unique students** with 0 missing values.
- **Feature Pruning:** Retained 21 early demographic, family, academic, and lifestyle predictors (excluded `G1` and `G2` to prevent data leakage).
- **Target Balance:** 84.29% Not At Risk (558) vs. 15.71% At Risk (104), handled via stratified 80/20 splitting and `class_weight="balanced"`.

## Model Benchmark (Test Set — 133 Students)
| Model | Accuracy | Precision (At Risk) | Recall (At Risk) | F1-Score (At Risk) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Deployed)** | 76.69% | 36.84% | **66.67%** | **0.4746** |
| Decision Tree (`max_depth=5`) | 70.68% | 28.57% | 57.14% | 0.3810 |
| Random Forest (`n_estimators=200`) | 84.96% | 53.33% | 38.10% | 0.4444 |

Logistic Regression was selected for deployment because maximizing **At Risk Recall (14/21 caught)** minimizes missed early interventions.

## Run Locally
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
