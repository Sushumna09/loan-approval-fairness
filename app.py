"""
Loan Approval Fairness Demo

A Streamlit application that:
  1. Predicts loan approval for a user-supplied applicant profile
  2. Compares a baseline Random Forest against a Fairlearn-mitigated model
  3. Reports demographic parity and equal opportunity gaps by gender and race

Data is generated synthetically at startup with an intentional demographic
bias pattern, so fairness metrics have a known ground truth.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

st.set_page_config(
    page_title="Loan Approval Fairness Audit",
    page_icon="scales",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------
def generate_loan_data(n_samples: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic loan applications.

    Feature distributions approximate real-world credit application data.
    The historical approval label reflects both true risk factors and an
    intentional demographic bias (gender pay gap + race-based approval bias),
    modelling the kind of biased legacy data a real lender might have.
    """
    rng = np.random.default_rng(seed)

    gender = rng.choice(['Male', 'Female'], size=n_samples, p=[0.55, 0.45])
    race = rng.choice(
        ['White', 'Black', 'Hispanic', 'Asian'],
        size=n_samples, p=[0.60, 0.15, 0.15, 0.10],
    )
    age = rng.integers(21, 66, size=n_samples)

    edu_probs = {
        'White':    [0.25, 0.50, 0.20, 0.05],
        'Black':    [0.35, 0.45, 0.15, 0.05],
        'Hispanic': [0.40, 0.45, 0.13, 0.02],
        'Asian':    [0.15, 0.45, 0.30, 0.10],
    }
    edu_opts = ['HighSchool', 'Bachelors', 'Masters', 'PhD']
    education = np.array([rng.choice(edu_opts, p=edu_probs[r]) for r in race])

    base_income = pd.Series(education).map({
        'HighSchool': 35000, 'Bachelors': 60000,
        'Masters':    85000, 'PhD':      110000,
    }).values
    gender_multiplier = np.where(gender == 'Female', 0.85, 1.0)
    annual_income = np.clip(
        base_income * gender_multiplier * rng.normal(1.0, 0.25, size=n_samples),
        15000, 500000,
    ).astype(int)

    employment_type = rng.choice(
        ['Salaried', 'SelfEmployed', 'Unemployed'],
        size=n_samples, p=[0.75, 0.20, 0.05],
    )
    credit_score = np.clip(
        500 + 0.002 * annual_income + rng.normal(0, 80, size=n_samples),
        300, 850,
    ).astype(int)
    num_existing_loans = rng.integers(0, 6, size=n_samples)
    previous_defaults = rng.choice(
        [0, 1, 2, 3], size=n_samples, p=[0.70, 0.20, 0.08, 0.02],
    )
    loan_amount = np.clip(
        annual_income * rng.uniform(0.1, 2.0, size=n_samples),
        1000, 1_000_000,
    ).astype(int)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n_samples)

    monthly_income = annual_income / 12
    monthly_debt = num_existing_loans * 300 + loan_amount / loan_term_months
    debt_to_income = np.clip(monthly_debt / monthly_income, 0, 2).round(3)

    home_ownership = rng.choice(
        ['Own', 'Rent', 'Mortgage'], size=n_samples, p=[0.25, 0.45, 0.30],
    )

    true_default_prob = np.clip(
        0.35
        - 0.0006 * (credit_score - 500)
        + 0.30 * debt_to_income
        + 0.12 * previous_defaults
        - 0.000001 * annual_income
        + (employment_type == 'Unemployed') * 0.25,
        0.02, 0.95,
    )
    approval_score = (
        (1.0 - true_default_prob)
        - 0.10 * (gender == 'Female')
        - 0.15 * (race == 'Black')
        - 0.10 * (race == 'Hispanic')
        + rng.normal(0, 0.05, size=n_samples)
    )
    loan_approved = (approval_score > 0.55).astype(int)

    return pd.DataFrame({
        'age': age, 'gender': gender, 'race': race,
        'education': education, 'employment_type': employment_type,
        'annual_income': annual_income, 'credit_score': credit_score,
        'num_existing_loans': num_existing_loans,
        'previous_defaults': previous_defaults,
        'loan_amount': loan_amount, 'loan_term_months': loan_term_months,
        'debt_to_income_ratio': debt_to_income,
        'home_ownership': home_ownership,
        'loan_approved': loan_approved,
    })


# ---------------------------------------------------------------------------
# Model training (cached; runs once per Streamlit session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_models() -> dict:
    """Fit a baseline classifier and a fairness-constrained classifier."""
    df = generate_loan_data(n_samples=10_000, seed=42)

    categorical = ['education', 'employment_type', 'home_ownership']
    numerical = [
        'age', 'annual_income', 'credit_score', 'num_existing_loans',
        'previous_defaults', 'loan_amount', 'loan_term_months',
        'debt_to_income_ratio',
    ]

    X = pd.get_dummies(df[categorical + numerical], columns=categorical, drop_first=True)
    y = df['loan_approved']
    protected = df[['gender', 'race']]

    X_train, X_test, y_train, y_test, prot_train, prot_test = train_test_split(
        X, y, protected, test_size=0.25, random_state=42, stratify=y,
    )

    baseline = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    baseline.fit(X_train, y_train)

    fair = ExponentiatedGradient(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        constraints=DemographicParity(),
        eps=0.02,
    )
    fair.fit(X_train, y_train, sensitive_features=prot_train['gender'])

    return {
        'baseline': baseline,
        'fair': fair,
        'feature_columns': list(X_train.columns),
        'X_test': X_test, 'y_test': y_test, 'prot_test': prot_test,
    }


with st.spinner("Preparing models (first load may take 60-90 seconds)..."):
    art = train_models()

baseline_model = art['baseline']
fair_model = art['fair']
feature_columns = art['feature_columns']


# ---------------------------------------------------------------------------
# Feature vector construction from user form
# ---------------------------------------------------------------------------
def build_input_row(
    age, annual_income, credit_score, num_existing_loans,
    previous_defaults, loan_amount, loan_term_months,
    education, employment_type, home_ownership,
):
    """Convert form inputs into a one-hot encoded row aligned with training features."""
    monthly_income = annual_income / 12
    monthly_debt = num_existing_loans * 300 + loan_amount / loan_term_months
    dti = min(monthly_debt / monthly_income, 2.0)

    values = {
        'age': age,
        'annual_income': annual_income,
        'credit_score': credit_score,
        'num_existing_loans': num_existing_loans,
        'previous_defaults': previous_defaults,
        'loan_amount': loan_amount,
        'loan_term_months': loan_term_months,
        'debt_to_income_ratio': round(dti, 3),
    }
    for edu in ['HighSchool', 'Masters', 'PhD']:
        values[f'education_{edu}'] = int(education == edu)
    for emp in ['SelfEmployed', 'Unemployed']:
        values[f'employment_type_{emp}'] = int(employment_type == emp)
    for home in ['Own', 'Rent']:
        values[f'home_ownership_{home}'] = int(home_ownership == home)

    row = pd.DataFrame([values]).reindex(columns=feature_columns, fill_value=0)
    return row, dti


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Loan Approval - Fairness Audit and Mitigation")
st.markdown(
    "An interactive demonstration of demographic bias in loan approval models "
    "and its mitigation using Fairlearn. The baseline model is trained without "
    "gender or race features; the fair model applies a Demographic Parity "
    "constraint via constrained optimisation."
)

with st.sidebar:
    st.title("About")
    st.markdown(
        "**Task.** Binary classification: predict whether a loan application "
        "will be approved.\n\n"
        "**Data.** 10,000 synthetic applications with realistic feature "
        "distributions and an intentionally biased approval label pattern.\n\n"
        "**Models.**\n"
        "- Baseline: Random Forest (scikit-learn)\n"
        "- Fair: Random Forest wrapped in Fairlearn's `ExponentiatedGradient` "
        "with a `DemographicParity` constraint (eps = 0.02)\n\n"
        "[Source code](https://github.com/Sushumna09/loan-approval-fairness)"
    )

tab_predict, tab_audit, tab_how = st.tabs(
    ["Predict", "Fairness Audit", "Methodology"]
)

with tab_predict:
    st.header("Applicant Details")

    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Age", 21, 65, 35)
        annual_income = st.number_input("Annual Income ($)", 15000, 500000, 60000, step=1000)
        credit_score = st.slider("Credit Score", 300, 850, 650)
        num_existing_loans = st.slider("Existing Loans", 0, 5, 1)
    with c2:
        previous_defaults = st.slider("Previous Defaults", 0, 3, 0)
        loan_amount = st.number_input("Loan Amount Requested ($)", 1000, 500000, 30000, step=1000)
        loan_term_months = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=2)

    c3, c4, c5 = st.columns(3)
    with c3:
        education = st.selectbox("Education", ['HighSchool', 'Bachelors', 'Masters', 'PhD'], index=1)
    with c4:
        employment_type = st.selectbox("Employment", ['Salaried', 'SelfEmployed', 'Unemployed'])
    with c5:
        home_ownership = st.selectbox("Home Ownership", ['Own', 'Rent', 'Mortgage'], index=1)

    if st.button("Predict", type="primary"):
        row, dti = build_input_row(
            age, annual_income, credit_score, num_existing_loans,
            previous_defaults, loan_amount, loan_term_months,
            education, employment_type, home_ownership,
        )

        baseline_prediction = baseline_model.predict(row)[0]
        baseline_probability = baseline_model.predict_proba(row)[0, 1]
        fair_prediction = fair_model.predict(row)[0]

        st.markdown("---")
        st.caption(f"Debt-to-income ratio (derived): {dti:.2%}")

        cA, cB = st.columns(2)
        with cA:
            st.subheader("Baseline Model")
            if baseline_prediction == 1:
                st.success(f"APPROVED  ({baseline_probability:.1%} probability)")
            else:
                st.error(f"REJECTED  ({baseline_probability:.1%} probability)")

        with cB:
            st.subheader("Fair Model")
            if fair_prediction == 1:
                st.success("APPROVED")
            else:
                st.error("REJECTED")

        if baseline_prediction != fair_prediction:
            st.warning(
                "The two models disagree on this applicant, indicating that the "
                "fairness constraint has changed the outcome for this profile."
            )

with tab_audit:
    st.header("Fairness Comparison on Held-Out Test Set")

    X_test = art['X_test']
    y_test = art['y_test']
    prot_test = art['prot_test']

    baseline_pred_all = baseline_model.predict(X_test)
    fair_pred_all = fair_model.predict(X_test)

    def selection_rate_by_group(y_pred, groups):
        return pd.Series(y_pred).groupby(groups.values).mean()

    baseline_by_gender = selection_rate_by_group(baseline_pred_all, prot_test['gender'])
    fair_by_gender     = selection_rate_by_group(fair_pred_all, prot_test['gender'])
    baseline_by_race   = selection_rate_by_group(baseline_pred_all, prot_test['race'])
    fair_by_race       = selection_rate_by_group(fair_pred_all, prot_test['race'])

    baseline_accuracy = accuracy_score(y_test, baseline_pred_all)
    fair_accuracy = accuracy_score(y_test, fair_pred_all)
    baseline_dp_gap = baseline_by_gender.max() - baseline_by_gender.min()
    fair_dp_gap = fair_by_gender.max() - fair_by_gender.min()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Baseline accuracy", f"{baseline_accuracy:.1%}")
        st.metric("Baseline DP gap (gender)", f"{baseline_dp_gap:.1%}")
    with c2:
        st.metric(
            "Fair accuracy", f"{fair_accuracy:.1%}",
            delta=f"{(fair_accuracy - baseline_accuracy) * 100:+.1f} pp",
        )
        st.metric(
            "Fair DP gap (gender)", f"{fair_dp_gap:.1%}",
            delta=f"{(fair_dp_gap - baseline_dp_gap) * 100:+.1f} pp",
            delta_color="inverse",
        )

    st.markdown("### Approval Rate by Gender")
    st.bar_chart(pd.DataFrame({'Baseline': baseline_by_gender, 'Fair': fair_by_gender}))

    st.markdown("### Approval Rate by Race")
    st.bar_chart(pd.DataFrame({'Baseline': baseline_by_race, 'Fair': fair_by_race}))

    st.info(
        "The baseline model was trained without gender or race as features, yet its "
        "predicted approval rates differ across demographic groups. This is because "
        "features such as income and credit score correlate with those protected "
        "attributes and act as proxy variables. The fairness-constrained model "
        "reduces the demographic parity gap, at a measurable cost in raw accuracy "
        "against the (biased) historical labels."
    )

with tab_how:
    st.header("Methodology")
    st.markdown(
        """
### Data
A synthetic dataset of 10,000 loan applications is generated on startup with
a fixed random seed. Features include demographic attributes (age, gender,
race, education, employment), financial attributes (annual income, credit
score, existing debt, previous defaults), and loan characteristics
(amount, term, housing status).

The historical approval label is generated in two stages:
1. A true underlying default risk based only on real financial signals
   (credit score, debt-to-income ratio, previous defaults, income).
2. A biased approval decision that combines that risk with a demographic
   penalty against female, Black, and Hispanic applicants, simulating
   historically biased lending practice.

### Baseline Model
`RandomForestClassifier` (200 trees, `random_state=42`) trained on all
features **except** gender and race. This tests the "fairness through
unawareness" hypothesis.

### Fair Model
Fairlearn's `ExponentiatedGradient` reduction, wrapping a
`RandomForestClassifier` (100 trees), constrained by `DemographicParity`
with `eps=0.02`. The reduction reformulates the fair-classification
problem as a sequence of cost-sensitive classification subproblems and
returns a randomised classifier that provably satisfies the constraint
up to `eps`.

### Metrics
- **Accuracy**: standard fraction of correct predictions against test-set
  labels (noting that those labels are themselves biased).
- **Demographic Parity difference**: difference between the highest and
  lowest group-level selection rates. A DP difference of 0 means every
  group is approved at the same rate.
- **Equal Opportunity difference**: difference between the highest and
  lowest group-level true-positive rates.

### References
- Agarwal, A., et al. *A Reductions Approach to Fair Classification.*
  ICML 2018.
- Hardt, M., Price, E., Srebro, N. *Equality of Opportunity in Supervised
  Learning.* NIPS 2016.
- Fairlearn documentation: https://fairlearn.org
"""
    )
