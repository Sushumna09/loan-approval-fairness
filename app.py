"""
Loan Approval Fairness Demo
Streamlit app: predict + audit bias + mitigate

Deploy to Hugging Face Spaces (Streamlit SDK).
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

st.set_page_config(page_title="Loan Fairness Demo",
                   page_icon="scales", layout="wide")


# ============================================================
# DATA GENERATION (same as notebook Cell 1)
# ============================================================
def generate_loan_data(n_samples=10000, seed=42):
    rng = np.random.default_rng(seed)
    gender = rng.choice(['Male', 'Female'], size=n_samples, p=[0.55, 0.45])
    race = rng.choice(['White', 'Black', 'Hispanic', 'Asian'],
                      size=n_samples, p=[0.60, 0.15, 0.15, 0.10])
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
        'Masters': 85000, 'PhD': 110000
    }).values
    gender_mult = np.where(gender == 'Female', 0.85, 1.0)
    annual_income = np.clip(
        base_income * gender_mult * rng.normal(1.0, 0.25, size=n_samples),
        15000, 500000
    ).astype(int)

    employment_type = rng.choice(
        ['Salaried', 'SelfEmployed', 'Unemployed'],
        size=n_samples, p=[0.75, 0.20, 0.05]
    )
    credit_score = np.clip(
        500 + 0.002 * annual_income + rng.normal(0, 80, size=n_samples),
        300, 850
    ).astype(int)
    num_existing_loans = rng.integers(0, 6, size=n_samples)
    previous_defaults = rng.choice([0, 1, 2, 3], size=n_samples,
                                   p=[0.70, 0.20, 0.08, 0.02])
    loan_amount = np.clip(
        annual_income * rng.uniform(0.1, 2.0, size=n_samples),
        1000, 1_000_000
    ).astype(int)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n_samples)

    monthly_income = annual_income / 12
    monthly_debt = num_existing_loans * 300 + loan_amount / loan_term_months
    dti = np.clip(monthly_debt / monthly_income, 0, 2).round(3)

    home_ownership = rng.choice(['Own', 'Rent', 'Mortgage'],
                                size=n_samples, p=[0.25, 0.45, 0.30])

    true_default_prob = np.clip(
        0.35
        - 0.0006 * (credit_score - 500)
        + 0.30 * dti
        + 0.12 * previous_defaults
        - 0.000001 * annual_income
        + (employment_type == 'Unemployed') * 0.25,
        0.02, 0.95
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
        'debt_to_income_ratio': dti, 'home_ownership': home_ownership,
        'loan_approved': loan_approved,
    })


# ============================================================
# TRAIN MODELS (cached across sessions)
# ============================================================
@st.cache_resource(show_spinner=False)
def train_models():
    df = generate_loan_data(n_samples=10000, seed=42)
    CAT = ['education', 'employment_type', 'home_ownership']
    NUM = ['age', 'annual_income', 'credit_score', 'num_existing_loans',
           'previous_defaults', 'loan_amount', 'loan_term_months',
           'debt_to_income_ratio']

    X = pd.get_dummies(df[CAT + NUM], columns=CAT, drop_first=True)
    y = df['loan_approved']
    protected = df[['gender', 'race']]

    X_tr, X_te, y_tr, y_te, p_tr, p_te = train_test_split(
        X, y, protected, test_size=0.25, random_state=42, stratify=y
    )

    baseline = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    baseline.fit(X_tr, y_tr)

    fair = ExponentiatedGradient(
        estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        constraints=DemographicParity(),
        eps=0.02,
    )
    fair.fit(X_tr, y_tr, sensitive_features=p_tr['gender'])

    return {
        'baseline': baseline, 'fair': fair,
        'feature_columns': list(X_tr.columns),
        'X_test': X_te, 'y_test': y_te, 'prot_test': p_te,
        'df': df,
    }


with st.spinner("Loading models (30-90 seconds on first load, cached after)..."):
    art = train_models()

baseline_model = art['baseline']
fair_model = art['fair']
feature_columns = art['feature_columns']


# ============================================================
# HELPER: build feature vector from user form
# ============================================================
def build_input_row(age, annual_income, credit_score, num_existing_loans,
                    previous_defaults, loan_amount, loan_term_months,
                    education, employment_type, home_ownership):
    monthly_income = annual_income / 12
    monthly_debt = num_existing_loans * 300 + loan_amount / loan_term_months
    dti = min(monthly_debt / monthly_income, 2.0)

    d = {
        'age': age,
        'annual_income': annual_income,
        'credit_score': credit_score,
        'num_existing_loans': num_existing_loans,
        'previous_defaults': previous_defaults,
        'loan_amount': loan_amount,
        'loan_term_months': loan_term_months,
        'debt_to_income_ratio': round(dti, 3),
    }
    for edu in ['HighSchool', 'Masters', 'PhD']:      # 'Bachelors' dropped
        d[f'education_{edu}'] = int(education == edu)
    for emp in ['SelfEmployed', 'Unemployed']:         # 'Salaried' dropped
        d[f'employment_type_{emp}'] = int(employment_type == emp)
    for home in ['Own', 'Rent']:                       # 'Mortgage' dropped
        d[f'home_ownership_{home}'] = int(home_ownership == home)

    row = pd.DataFrame([d]).reindex(columns=feature_columns, fill_value=0)
    return row, dti


# ============================================================
# UI
# ============================================================
st.title("Loan Approval — Fairness Audit Demo")
st.markdown(
    "A loan approval classifier with **demographic bias detection + mitigation** using "
    "Fairlearn. See how a naive ML model discriminates, and how targeted mitigation "
    "fixes it — without ever using gender/race as a feature."
)

with st.sidebar:
    st.title("About")
    st.markdown("""
Built on **synthetic data** with intentional bias by gender and race.

**Comparison:**
- Baseline Random Forest (inherits bias)
- Fairlearn-mitigated model (bias reduced)

**Tech:** scikit-learn · Fairlearn · Streamlit · Hugging Face Spaces

[Source code on GitHub](https://github.com/Sushumna09/loan-approval-fairness)
""")

tab_predict, tab_audit, tab_how = st.tabs(
    ["Predict", "Fairness Audit", "How It Works"]
)

# ---- TAB 1: PREDICT ----
with tab_predict:
    st.header("Enter Applicant Details")

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
            education, employment_type, home_ownership
        )
        b_pred = baseline_model.predict(row)[0]
        b_proba = baseline_model.predict_proba(row)[0, 1]
        f_pred = fair_model.predict(row)[0]

        st.markdown("---")
        st.caption(f"Debt-to-income ratio (derived): {dti:.2%}")
        cA, cB = st.columns(2)
        with cA:
            st.subheader("Baseline Model")
            st.caption("Trained without gender/race, but inherits bias via proxies")
            if b_pred == 1:
                st.success(f"APPROVED   ({b_proba:.1%} probability)")
            else:
                st.error(f"REJECTED   ({b_proba:.1%} probability)")
        with cB:
            st.subheader("Fair Model (Fairlearn)")
            st.caption("Same data, with Demographic Parity constraint")
            if f_pred == 1:
                st.success("APPROVED")
            else:
                st.error("REJECTED")

        if b_pred != f_pred:
            st.warning(
                "**The two models disagree on this applicant.** "
                "This is where fairness mitigation actively changes an outcome."
            )

# ---- TAB 2: AUDIT ----
with tab_audit:
    st.header("Before vs After Bias Mitigation")
    st.markdown("Measured on the held-out test set.")

    X_test = art['X_test']
    y_test = art['y_test']
    prot_test = art['prot_test']

    b_pred_all = baseline_model.predict(X_test)
    f_pred_all = fair_model.predict(X_test)

    def group_sel(y_pred, groups):
        return pd.Series(y_pred).groupby(groups.values).mean()

    b_g = group_sel(b_pred_all, prot_test['gender'])
    f_g = group_sel(f_pred_all, prot_test['gender'])
    b_r = group_sel(b_pred_all, prot_test['race'])
    f_r = group_sel(f_pred_all, prot_test['race'])

    b_acc = accuracy_score(y_test, b_pred_all)
    f_acc = accuracy_score(y_test, f_pred_all)
    b_dp = b_g.max() - b_g.min()
    f_dp = f_g.max() - f_g.min()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Baseline accuracy", f"{b_acc:.1%}")
        st.metric("Baseline DP gap (gender)", f"{b_dp:.1%}")
    with c2:
        st.metric("Fair accuracy", f"{f_acc:.1%}",
                  delta=f"{(f_acc - b_acc)*100:+.1f}pp")
        st.metric("Fair DP gap (gender)", f"{f_dp:.1%}",
                  delta=f"{(f_dp - b_dp)*100:+.1f}pp",
                  delta_color="inverse")

    st.markdown("### Approval Rates by Gender")
    st.bar_chart(pd.DataFrame({'Baseline': b_g, 'Fair': f_g}))

    st.markdown("### Approval Rates by Race")
    st.bar_chart(pd.DataFrame({'Baseline': b_r, 'Fair': f_r}))

    st.info(
        "**Takeaway:** the baseline model was trained WITHOUT gender/race, "
        "yet its predicted approval rates differ by demographic. "
        "Correlated features (income ↔ gender, credit ↔ race) act as proxies. "
        "Fairlearn's Exponentiated Gradient with a Demographic Parity constraint "
        "shrinks the gaps."
    )

# ---- TAB 3: HOW IT WORKS ----
with tab_how:
    st.header("Project Overview")
    st.markdown("""
### The Problem
Loan approval models trained on historical data often inherit bias against women
and minorities — even when protected attributes (gender, race) are dropped from
the training features. This is because other features (income, credit score,
education) *correlate* with those attributes.

### The Approach
1. **Generate synthetic data** with intentional demographic bias
2. **Train a baseline Random Forest** without gender/race features
3. **Measure bias**: demographic parity + equal opportunity across groups
4. **Mitigate**: apply Fairlearn's Exponentiated Gradient (Agarwal et al., 2018)
5. **Explain**: use SHAP to show which features drive predictions

### Tech Stack
scikit-learn · Fairlearn · SHAP · Streamlit · Hugging Face Spaces

### Source Code
https://github.com/Sushumna09/loan-approval-fairness

### Why Synthetic Data?
So we know exactly where the bias lives and can measure whether our fairness
methods correctly detect and remove it. Same approach as IBM AIF360 and other
fairness research benchmarks.
""")
