# Loan Approval - Fairness Audit and Mitigation

An end-to-end machine learning project that detects and mitigates demographic bias in a loan approval classifier. Built on synthetic data with a controlled bias pattern, the project demonstrates a full fairness workflow: audit, mitigation, and explainability, deployed as an interactive web application.

## Overview

Standard machine learning models trained on historical lending data often reproduce demographic bias present in that data - even when protected attributes such as gender and race are explicitly withheld from the training features. This is because non-protected features (income, credit score, education) correlate with protected ones and act as proxy variables.

This project:
- Generates a synthetic dataset with a known bias pattern to enable ground-truth evaluation of fairness methods.
- Trains a baseline Random Forest classifier without protected attributes and shows that it still discriminates.
- Applies Fairlearn's `ExponentiatedGradient` reduction with a Demographic Parity constraint to mitigate the bias.
- Uses SHAP to attribute predictions to input features and expose the proxy-variable mechanism.
- Deploys the trained models and audit dashboard as a Streamlit application.

## Live Demo

*(Add your Streamlit Community Cloud URL here after deployment)*

## Results

Evaluated on a held-out 25% test split of the 10,000-sample synthetic dataset:

| Metric | Baseline Random Forest | Fairlearn-constrained | Change |
|---|---|---|---|
| Accuracy | 84.0% | 72.4% | -11.6 pp |
| Demographic Parity gap (gender) | 4.8% | 2.7% | -2.1 pp |
| Equal Opportunity gap (gender) | 17.2% | 11.4% | -5.8 pp |

Accuracy is measured against the (biased) historical labels; a fair model necessarily disagrees with those labels on a subset of decisions, so a reduction in observed accuracy is expected. Evaluating against an unbiased holdout would be preferred in production but requires an independent labelling process outside the scope of this demonstration.

## Methodology

### Data Generation
A synthetic dataset of 10,000 loan applications is produced with a fixed random seed. Features include age, gender, race, education, employment type, annual income, credit score, existing loans, previous defaults, loan amount, loan term, debt-to-income ratio, and home ownership. Income is generated conditional on education and gender (embedding a pay gap). Historical approval labels are produced from two components:

1. A true underlying default probability based only on financial signals.
2. A demographic penalty applied against female, Black, and Hispanic applicants that shifts approval decisions below their true risk profile.

### Baseline Model
A `RandomForestClassifier` (200 trees) trained on all features excluding gender and race. This baseline tests the "fairness through unawareness" strategy and demonstrates its failure: the model still produces demographically disparate outcomes because other features correlate with the withheld attributes.

### Fair Model
Fairlearn's `ExponentiatedGradient` reduction wraps a `RandomForestClassifier` (100 trees) with a `DemographicParity` constraint (tolerance eps = 0.02). The reduction reformulates the fair classification problem as a sequence of cost-sensitive classification subproblems and returns a randomised classifier that satisfies the parity constraint up to eps.

### Fairness Metrics
- **Selection rate**: proportion of applicants in a group receiving a positive prediction.
- **Demographic Parity difference**: max - min selection rate across groups.
- **True Positive Rate**: proportion of truly-approved applicants correctly identified.
- **Equal Opportunity difference**: max - min TPR across groups.

### Explainability
SHAP TreeExplainer computes feature attributions on a random sample of the test set. Global feature importance identifies which features carry the most predictive signal; the resulting ranking shows income and credit score as top drivers, confirming they act as the primary proxies for the withheld demographic attributes.

## Tech Stack

| Layer | Library |
|---|---|
| Data and modelling | scikit-learn, NumPy, pandas |
| Fairness | Fairlearn |
| Explainability | SHAP |
| Application | Streamlit |
| Hosting | Streamlit Community Cloud |

## Local Setup

```bash
git clone https://github.com/Sushumna09/loan-approval-fairness.git
cd loan-approval-fairness
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in a browser.

## Repository Layout

```
loan-approval-fairness/
├── app.py                # Streamlit application (self-contained)
├── requirements.txt      # Python dependencies
├── notebooks/            # Analysis notebooks
├── .gitignore
└── README.md
```

## References

- Agarwal, A., Beygelzimer, A., Dudik, M., Langford, J., Wallach, H. *A Reductions Approach to Fair Classification.* ICML 2018.
- Hardt, M., Price, E., Srebro, N. *Equality of Opportunity in Supervised Learning.* NIPS 2016.
- Lundberg, S., Lee, S. *A Unified Approach to Interpreting Model Predictions.* NIPS 2017.
- [Fairlearn documentation](https://fairlearn.org)

## Author

Sushumna Devi Gajarla - [GitHub](https://github.com/Sushumna09)

## License

MIT
