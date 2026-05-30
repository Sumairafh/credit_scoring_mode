# 💳 Credit Scoring Model

> An end-to-end Machine Learning project that predicts whether a loan applicant is a **Good** or **Bad** credit risk — from raw data to a production-ready Streamlit web application.

---

## 📋 Project Overview

Banks and lending institutions need to assess the creditworthiness of loan applicants quickly and accurately. This project builds a complete ML pipeline that:

- Ingests and cleans historical applicant data
- Engineers informative financial ratios
- Trains and compares three classifiers
- Selects the best model automatically
- Serves predictions via a polished Streamlit web app

---

## 🗂️ Folder Structure

```
credit_scoring_model/
│
├── data/
│   ├── raw/
│   │   ├── credit_data.csv          # Raw dataset (2000 rows)
│   │   └── generate_data.py         # Script to regenerate dataset
│   └── processed/                   # Created after preprocessing
│
├── notebooks/
│   └── EDA.ipynb                    # Exploratory Data Analysis
│
├── models/
│   └── credit_model.pkl             # Saved best model (after training)
│
├── src/
│   ├── utils.py                     # Shared helpers & path utilities
│   ├── data_preprocessing.py        # Clean, scale, split data
│   ├── feature_engineering.py       # Derive financial ratios
│   ├── train_model.py               # Train + select best model
│   ├── evaluate_model.py            # Metrics, confusion matrix, ROC
│   └── predict.py                   # Load model & make predictions
│
├── app/
│   └── app.py                       # Streamlit web application
│
├── reports/
│   ├── confusion_matrix.png         # Created after training
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation & Setup

### 1. Clone / Download the project

```bash
git clone https://github.com/yourname/credit_scoring_model.git
cd credit_scoring_model
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏃 How to Run

### Step 1 — Train the model

```bash
python src/train_model.py
```

This will:
- Load `data/raw/credit_data.csv`
- Preprocess + engineer features
- Train Logistic Regression, Decision Tree, and Random Forest
- Pick the best model by ROC-AUC
- Save it to `models/credit_model.pkl`
- Generate plots in `reports/`

### Step 2 — Make a single prediction (CLI)

```bash
python src/predict.py
```

Edit the `sample` dict inside `predict.py` to test different applicants.

### Step 3 — Launch the web app

```bash
streamlit run app/app.py
```

Open your browser at **http://localhost:8501**

### Step 4 — Explore the EDA notebook

```bash
jupyter notebook notebooks/EDA.ipynb
```

---

## 📊 Dataset Description

The dataset contains **2,000 synthetic applicant records** with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `age` | int | Applicant age (years) |
| `income` | int | Annual income (USD) |
| `employment_years` | int | Years at current employer |
| `debt` | int | Total outstanding debt (USD) |
| `loan_amount` | int | Requested loan amount (USD) |
| `credit_history_years` | int | Length of credit history |
| `number_of_credit_cards` | int | Active credit cards |
| `payment_history_score` | int | Credit bureau score (300–850) |
| `missed_payments` | int | Total missed payments |
| `credit_utilization` | float | Credit used ÷ total available |
| `target` | int | **0** = Bad, **1** = Good credit risk |

### Engineered Features

| Feature | Formula |
|---------|---------|
| `debt_to_income_ratio` | debt ÷ income |
| `loan_to_income_ratio` | loan_amount ÷ income |
| `credit_card_per_income` | cards ÷ (income / 10k) |
| `payment_reliability_score` | normalised payment score − missed penalty |
| `total_obligation_ratio` | (debt + loan) ÷ income |
| `credit_experience` | log(credit_years + 1) + log(employment_years + 1) |

---

## 🤖 Algorithms Used

| Model | Key Hyperparameters |
|-------|-------------------|
| **Logistic Regression** | `max_iter=1000`, `class_weight='balanced'` |
| **Decision Tree** | `max_depth=6`, `min_samples_leaf=10` |
| **Random Forest** | `n_estimators=200`, `max_depth=8`, `n_jobs=-1` |

All models use `class_weight='balanced'` to handle class imbalance without oversampling.

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted "Good", how many truly are? |
| **Recall** | Of truly "Good", how many did we catch? |
| **F1 Score** | Harmonic mean of precision & recall |
| **ROC-AUC** | Model's ability to discriminate classes |

The model with the highest **ROC-AUC** is automatically selected.

---

## 🖼️ Screenshots

> After training, PNG files are saved to `reports/`:

- `reports/confusion_matrix.png` — true vs predicted labels
- `reports/roc_curve.png` — ROC curve with AUC score
- `reports/feature_importance.png` — most influential features

The **Streamlit app** shows:
- Credit risk verdict (Good / Bad) with colour-coded banner
- Probability gauge bar
- Risk level badge (Low / Medium / High)
- Derived financial ratios at a glance

---

## 🔭 Future Improvements

- [ ] Hyperparameter tuning with `GridSearchCV` or `Optuna`
- [ ] Add XGBoost / LightGBM models for better performance
- [ ] SHAP explainability charts for individual predictions
- [ ] Upload CSV for batch predictions in the web app
- [ ] Docker container for one-command deployment
- [ ] REST API using FastAPI for integration with other services
- [ ] Real dataset integration (e.g. UCI German Credit dataset)
- [ ] Automated re-training pipeline when new data arrives

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Core language |
| pandas | 2.0+ | Data manipulation |
| NumPy | 1.24+ | Numerical computing |
| scikit-learn | 1.3+ | ML models & preprocessing |
| Matplotlib | 3.7+ | Plotting |
| Seaborn | 0.12+ | Statistical visualisation |
| Joblib | 1.3+ | Model serialisation |
| Streamlit | 1.28+ | Web application |
| Jupyter | 7.0+ | EDA notebook |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built with ❤️ using Python & scikit-learn*
