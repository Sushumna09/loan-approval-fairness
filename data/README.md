# Dataset — Home Credit Default Risk

**Source**: [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data)

## Download Instructions

### Option 1: Manual Download (easiest)

1. Sign in to Kaggle
2. Go to https://www.kaggle.com/c/home-credit-default-risk/data
3. Accept the competition rules (one-click)
4. Download `application_train.csv` (this is the main file we'll use)
5. Move it into this `data/` folder

For the scope of this project, only `application_train.csv` is needed.

Final structure should look like:

```
data/
├── README.md
└── application_train.csv
```

### Option 2: Kaggle API (recommended for reproducibility)

```bash
pip install kaggle

# Get your API token from https://www.kaggle.com/settings/account
# Download kaggle.json, then:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Now download the dataset (from project root)
cd data/
kaggle competitions download -c home-credit-default-risk -f application_train.csv
unzip application_train.csv.zip
rm application_train.csv.zip
```

## Dataset Info

| Property | Value |
|---|---|
| Rows | 307,511 loan applications |
| Target | `TARGET` (1 = default, 0 = repaid) |
| Default rate | ~8% (imbalanced) |
| Features | 122 columns |
| Size | ~166 MB |

## Key Features for This Project

### Target
- **TARGET**: 1 if client had payment difficulties, 0 otherwise

### Protected Attributes (used for fairness audit)
- **CODE_GENDER**: M / F / XNA
- **DAYS_BIRTH**: negative days since birth (we convert to age)

### Predictive Features (partial list)
- **AMT_INCOME_TOTAL**: Client income
- **AMT_CREDIT**: Loan amount
- **AMT_ANNUITY**: Loan annuity
- **NAME_EDUCATION_TYPE**: Education level
- **NAME_FAMILY_STATUS**: Marital status
- **NAME_INCOME_TYPE**: Working, Pensioner, etc.
- **DAYS_EMPLOYED**: Days employed
- **REGION_RATING_CLIENT**: Region rating (1-3)
- **EXT_SOURCE_1/2/3**: External credit scores (0-1)

## Why This Dataset

- **Real-world scale** (307K applications)
- **Contains protected attributes** (gender, age) required for fairness auditing
- **Imbalanced target** (~8% default rate) teaches proper handling
- **Rich feature set** (122 columns) — realistic modeling challenge
- **Industry-standard** — actively used by Home Credit Group in production
