# Loan Approval Prediction with Fairness & Bias Analysis

End-to-end ML system that predicts loan default risk **and** audits itself for demographic bias. Because a model that's 84% accurate but discriminates against half the population isn't a good model.

## Why This Project

Loan approval ML models routinely embed historical biases (gender, age, race) present in training data. Regulators like the **RBI (India)** and **CFPB (US)** now require lenders to audit AI models for demographic fairness. This project demonstrates:

1. Building a competitive loan default predictor (XGBoost)
2. Auditing it for gender and age bias using **Fairlearn**
3. Applying mitigation techniques to reduce bias
4. Explaining individual predictions via **SHAP**
5. Wrapping everything in a Streamlit dashboard

## Core Question

> "Does my model give women loans at the same rate as men **at equal credit scores**? If not — how do I fix it without destroying accuracy?"

## Tech Stack

| Layer | Tools |
|---|---|
| Data | pandas, NumPy |
| ML | scikit-learn, XGBoost |
| Fairness | Fairlearn |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Deployment | Hugging Face Spaces |

## Dataset

**Home Credit Default Risk** (Kaggle) — 307K loan applications with gender, age, income, employment, credit history features.

See [`data/README.md`](data/README.md) for download instructions.

## Setup

```bash
git clone https://github.com/Sushumna09/loan-approval-fairness.git
cd loan-approval-fairness

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

jupyter lab
```

## Project Structure

```
loan-approval-fairness/
├── data/           # Dataset (not committed to git)
├── notebooks/      # Step-by-step Jupyter notebooks
├── src/            # Reusable Python modules
├── models/         # Trained model artifacts
└── app/            # Streamlit dashboard
```

## Roadmap

- [ ] **Week 1**: Python basics + EDA on Home Credit dataset
- [ ] **Week 2**: Baseline Logistic Regression + XGBoost models
- [ ] **Week 3**: Fairness audit (Fairlearn) + bias mitigation + SHAP
- [ ] **Week 4**: Streamlit dashboard + deploy on Hugging Face Spaces

## Results

_Will be updated as project progresses._

### Model Performance

| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | TBD | TBD | TBD | TBD |
| XGBoost (raw) | TBD | TBD | TBD | TBD |
| XGBoost (bias-mitigated) | TBD | TBD | TBD | TBD |

### Fairness Metrics by Gender

| Model | Demographic Parity Gap | Equal Opportunity Gap |
|---|---|---|
| XGBoost (raw) | TBD | TBD |
| XGBoost (mitigated) | TBD | TBD |

## Key Concepts Demonstrated

- Handling class imbalance in binary classification
- Fairness metrics: demographic parity, equalized odds, equal opportunity
- Bias mitigation: reweighing, threshold optimization
- Model explainability: global feature importance + individual SHAP explanations
- Responsible AI principles applied end-to-end

## Author

Sushumna Devi Gajarla — [GitHub](https://github.com/Sushumna09)

## License

MIT
