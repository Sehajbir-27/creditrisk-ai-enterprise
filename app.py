import math
import os
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "enterprise_credit_data.csv")


def calculate_emi(principal, annual_rate, months):
  if months <= 0:
    return 0.0
  if annual_rate <= 0:
    return principal / months
  r = annual_rate / (12.0 * 100.0)
  factor = math.pow(1 + r, months)
  return (principal * r * factor) / (factor - 1)


def generate_amortization(principal, annual_rate, months):
  r = annual_rate / (12.0 * 100.0)
  emi = calculate_emi(principal, annual_rate, months)
  balance = principal
  schedule = []
  total_years = math.ceil(months / 12)

  for year in range(1, total_years + 1):
    opening = balance
    yr_principal = 0.0
    yr_interest = 0.0
    months_in_yr = min(12, months - (year - 1) * 12)
    for _ in range(months_in_yr):
      if balance <= 0:
        break
      interest_comp = balance * r
      principal_comp = min(emi - interest_comp, balance)
      yr_interest += interest_comp
      yr_principal += principal_comp
      balance = max(0.0, balance - principal_comp)
    schedule.append({
        "year": year,
        "opening_balance": round(opening, 0),
        "principal_paid": round(yr_principal, 0),
        "interest_paid": round(yr_interest, 0),
        "closing_balance": round(balance, 0),
    })
  return schedule


def build_dataset(path, rows=2400):
  np.random.seed(42)
  app_inc = np.random.randint(22000, 280000, size=rows)
  co_inc = np.random.choice([0, 0, 15000, 35000, 60000, 90000], size=rows)
  existing_emi = np.random.choice([0, 0, 5000, 12000, 24000, 38000], size=rows)
  loan_amt = np.random.randint(150000, 6500000, size=rows)
  tenure = np.random.choice([24, 36, 60, 120, 180, 240, 300], size=rows)
  rate = np.round(np.random.uniform(8.2, 15.5, size=rows), 2)
  cibil = np.random.randint(510, 890, size=rows)
  emp_type = np.random.choice([0, 1, 2, 3], size=rows, p=[0.35, 0.35, 0.18, 0.12])
  years_job = np.random.randint(0, 22, size=rows)
  prop_area = np.random.choice([0, 1, 2], size=rows, p=[0.2, 0.35, 0.45])
  ltv_ratio = np.random.randint(45, 95, size=rows)

  emis = np.array(
      [calculate_emi(p, r, t) for p, r, t in zip(loan_amt, rate, tenure)]
  )
  tot_inc = app_inc + co_inc
  foir = ((existing_emi + emis) / tot_inc) * 100.0

  noise = np.random.normal(0, 10, size=rows)
  approved = (
      ((cibil + noise) >= 680)
      & (foir <= 52.0)
      & (ltv_ratio <= 85)
      & (tot_inc >= 25000)
      & ((years_job >= 1) | (emp_type == 0))
  ).astype(int)

  df = pd.DataFrame({
      "ApplicantIncome": app_inc,
      "CoapplicantIncome": co_inc,
      "ExistingEMI": existing_emi,
      "LoanAmount": loan_amt,
      "TenureMonths": tenure,
      "InterestRate": rate,
      "CIBIL_Score": cibil,
      "EmploymentType": emp_type,
      "YearsAtJob": years_job,
      "PropertyArea": prop_area,
      "LTV_Ratio": ltv_ratio,
      "FOIR_Percent": np.round(foir, 2),
      "Loan_Status": approved,
  })
  df.to_csv(path, index=False)
  return df


# --- Pure NumPy Machine Learning Suite ---


class CARTTree:

  def __init__(self, max_depth=5, min_samples=10, max_features=None, mode="clf"):
    self.max_depth = max_depth
    self.min_samples = min_samples
    self.max_features = max_features
    self.mode = mode
    self.tree = None
    self.importances = None

  def _build(self, X, y, depth):
    n_samples, n_features = X.shape
    val = float(np.mean(y)) if n_samples > 0 else 0.0

    if depth >= self.max_depth or n_samples < self.min_samples:
      return {"leaf": True, "val": val}
    if self.mode == "clf" and (val == 0.0 or val == 1.0):
      return {"leaf": True, "val": val}

    k = self.max_features if self.max_features else n_features
    feat_idx = np.random.choice(n_features, min(k, n_features), replace=False)

    parent_loss = (
        (2.0 * val * (1.0 - val)) if self.mode == "clf" else float(np.var(y))
    )
    best_gain = 0.0
    best_feat = None
    best_thresh = None

    for f in feat_idx:
      col = X[:, f]
      thresholds = np.unique(np.percentile(col, np.linspace(12, 88, 12)))
      if len(thresholds) == 0:
        continue

      left_mask = col[:, None] <= thresholds[None, :]
      n_left = left_mask.sum(axis=0)
      n_right = n_samples - n_left
      valid = (n_left >= 3) & (n_right >= 3)
      if not np.any(valid):
        continue

      left_mask = left_mask[:, valid]
      thresholds = thresholds[valid]
      n_left = n_left[valid]
      n_right = n_right[valid]

      sum_left = (y[:, None] * left_mask).sum(axis=0)
      sum_right = y.sum() - sum_left
      mean_left = sum_left / n_left
      mean_right = sum_right / n_right

      if self.mode == "clf":
        loss_left = 2.0 * mean_left * (1.0 - mean_left)
        loss_right = 2.0 * mean_right * (1.0 - mean_right)
      else:
        sq_sum = (y**2).sum()
        sq_left = ((y**2)[:, None] * left_mask).sum(axis=0)
        sq_right = sq_sum - sq_left
        loss_left = np.maximum(0.0, (sq_left / n_left) - (mean_left**2))
        loss_right = np.maximum(0.0, (sq_right / n_right) - (mean_right**2))

      weighted_loss = (n_left * loss_left + n_right * loss_right) / n_samples
      gains = parent_loss - weighted_loss
      idx = int(np.argmax(gains))
      if gains[idx] > best_gain:
        best_gain = float(gains[idx])
        best_feat = int(f)
        best_thresh = float(thresholds[idx])

    if best_feat is None:
      return {"leaf": True, "val": val}

    self.importances[best_feat] += best_gain * (n_samples / self.total_n)
    mask = X[:, best_feat] <= best_thresh
    return {
        "leaf": False,
        "feat": best_feat,
        "thresh": best_thresh,
        "left": self._build(X[mask], y[mask], depth + 1),
        "right": self._build(X[~mask], y[~mask], depth + 1),
    }

  def fit(self, X, y):
    self.total_n = len(X)
    self.importances = np.zeros(X.shape[1], dtype=float)
    self.tree = self._build(X, y, 0)

  def _pred_one(self, row, node):
    if node["leaf"]:
      return node["val"]
    if row[node["feat"]] <= node["thresh"]:
      return self._pred_one(row, node["left"])
    return self._pred_one(row, node["right"])

  def predict(self, X):
    return np.array([self._pred_one(r, self.tree) for r in X])


class RandomForestModel:

  def __init__(self, n_trees=35, max_depth=6, max_features=5, seed=42):
    self.n_trees = n_trees
    self.max_depth = max_depth
    self.max_features = max_features
    self.seed = seed
    self.trees = []
    self.feature_importances_ = None

  def fit(self, X, y):
    np.random.seed(self.seed)
    self.trees = []
    imp = np.zeros(X.shape[1], dtype=float)
    n = len(X)
    for _ in range(self.n_trees):
      idx = np.random.choice(n, n, replace=True)
      t = CARTTree(
          max_depth=self.max_depth, max_features=self.max_features, mode="clf"
      )
      t.fit(X[idx], y[idx])
      self.trees.append(t)
      imp += t.importances
    total = imp.sum()
    self.feature_importances_ = imp / total if total > 0 else imp

  def predict_proba(self, X):
    preds = np.array([t.predict(X) for t in self.trees])
    return np.mean(preds, axis=0)


class GradientBoostingModel:

  def __init__(self, n_estimators=22, lr=0.22, max_depth=3, seed=42):
    self.n_estimators = n_estimators
    self.lr = lr
    self.max_depth = max_depth
    self.seed = seed
    self.trees = []
    self.base_log_odds = 0.0

  def _sigmoid(self, z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

  def fit(self, X, y):
    np.random.seed(self.seed)
    p = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
    self.base_log_odds = float(np.log(p / (1.0 - p)))
    F = np.full(len(y), self.base_log_odds, dtype=float)
    self.trees = []

    for _ in range(self.n_estimators):
      probs = self._sigmoid(F)
      residuals = y - probs
      tree = CARTTree(max_depth=self.max_depth, mode="reg")
      tree.fit(X, residuals)
      F += self.lr * tree.predict(X)
      self.trees.append(tree)

  def predict_proba(self, X):
    F = np.full(len(X), self.base_log_odds, dtype=float)
    for tree in self.trees:
      F += self.lr * tree.predict(X)
    return self._sigmoid(F)


class LogisticRegressionModel:

  def __init__(self, lr=0.15, epochs=600, l2=0.05):
    self.lr = lr
    self.epochs = epochs
    self.l2 = l2
    self.w = None
    self.b = 0.0

  def _sigmoid(self, z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

  def fit(self, X, y):
    n, d = X.shape
    self.w = np.zeros(d, dtype=float)
    self.b = 0.0
    for _ in range(self.epochs):
      preds = self._sigmoid(X @ self.w + self.b)
      err = preds - y
      dw = (X.T @ err) / n + (self.l2 * self.w) / n
      db = float(np.mean(err))
      self.w -= self.lr * dw
      self.b -= self.lr * db

  def predict_proba(self, X):
    return self._sigmoid(X @ self.w + self.b)


def evaluate_metrics(y_true, y_prob):
  y_pred = (y_prob >= 0.5).astype(int)
  acc = float(np.mean(y_pred == y_true))
  tp = float(np.sum((y_pred == 1) & (y_true == 1)))
  fp = float(np.sum((y_pred == 1) & (y_true == 0)))
  fn = float(np.sum((y_pred == 0) & (y_true == 1)))
  prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

  pos = y_prob[y_true == 1]
  neg = y_prob[y_true == 0]
  if len(pos) > 0 and len(neg) > 0:
    auc = float(
        (
            np.sum(pos[:, None] > neg[None, :])
            + 0.5 * np.sum(pos[:, None] == neg[None, :])
        )
        / (len(pos) * len(neg))
    )
  else:
    auc = 0.5

  return {
      "accuracy": round(acc * 100, 2),
      "f1_score": round(f1 * 100, 2),
      "roc_auc": round(auc, 3),
  }


# --- Load Data & Train Multi-Model Suite ---

df = (
    build_dataset(CSV_PATH)
    if not os.path.exists(CSV_PATH)
    else pd.read_csv(CSV_PATH)
)

FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "ExistingEMI",
    "LoanAmount",
    "TenureMonths",
    "InterestRate",
    "CIBIL_Score",
    "EmploymentType",
    "YearsAtJob",
    "PropertyArea",
    "LTV_Ratio",
    "FOIR_Percent",
]

FEATURE_LABELS = [
    "Applicant Income",
    "Co-Applicant Income",
    "Existing EMIs",
    "Loan Amount",
    "Tenure (Months)",
    "Interest Rate",
    "CIBIL Score",
    "Employment Type",
    "Work Experience",
    "Property Area",
    "Loan-to-Value (LTV)",
    "FOIR / Debt Ratio",
]

X_all = df[FEATURES].to_numpy(dtype=float)
y_all = df["Loan_Status"].to_numpy(dtype=int)

np.random.seed(42)
perm = np.random.permutation(len(X_all))
split = int(0.8 * len(X_all))
X_train, y_train = X_all[perm[:split]], y_all[perm[:split]]
X_test, y_test = X_all[perm[split:]], y_all[perm[split:]]

mean_vec = X_train.mean(axis=0)
std_vec = np.where(X_train.std(axis=0) == 0, 1.0, X_train.std(axis=0))
X_train_s = (X_train - mean_vec) / std_vec
X_test_s = (X_test - mean_vec) / std_vec

rf_model = RandomForestModel(n_trees=35, max_depth=6, max_features=5)
rf_model.fit(X_train_s, y_train)

gbdt_model = GradientBoostingModel(n_estimators=22, lr=0.22, max_depth=3)
gbdt_model.fit(X_train_s, y_train)

lr_model = LogisticRegressionModel(lr=0.18, epochs=650, l2=0.05)
lr_model.fit(X_train_s, y_train)

p_rf = rf_model.predict_proba(X_test_s)
p_gb = gbdt_model.predict_proba(X_test_s)
p_lr = lr_model.predict_proba(X_test_s)
p_ens = 0.50 * p_rf + 0.35 * p_gb + 0.15 * p_lr

MODEL_BENCHMARKS = {
    "ensemble": evaluate_metrics(y_test, p_ens),
    "random_forest": evaluate_metrics(y_test, p_rf),
    "gradient_boosting": evaluate_metrics(y_test, p_gb),
    "logistic_regression": evaluate_metrics(y_test, p_lr),
}

FEATURE_IMPORTANCES = sorted(
    [
        {"feature": lbl, "importance": round(float(imp) * 100, 1)}
        for lbl, imp in zip(FEATURE_LABELS, rf_model.feature_importances_)
    ],
    key=lambda x: x["importance"],
    reverse=True,
)

print("\n" + "=" * 68)
print(" CREDITRISK AI ENTERPRISE â€” MULTI-MODEL ENSEMBLE ONLINE")
print(
    f" META-ENSEMBLE : Accuracy {MODEL_BENCHMARKS['ensemble']['accuracy']}%"
    f" | ROC-AUC {MODEL_BENCHMARKS['ensemble']['roc_auc']}"
)
print(" DASHBOARD URL : http://127.0.0.1:5000")
print("=" * 68 + "\n")


@app.route("/")
def serve_ui():
  return send_from_directory(BASE_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
  data = request.get_json()

  app_inc = float(data.get("ApplicantIncome", 85000))
  co_inc = float(data.get("CoapplicantIncome", 25000))
  existing_emi = float(data.get("ExistingEMI", 5000))
  loan_amt = float(data.get("LoanAmount", 1800000))
  tenure = int(data.get("TenureMonths", 180))
  rate = float(data.get("InterestRate", 9.2))
  cibil = int(data.get("CIBIL_Score", 780))
  emp_type = int(data.get("EmploymentType", 0))
  years_job = int(data.get("YearsAtJob", 4))
  prop_area = int(data.get("PropertyArea", 2))
  ltv_ratio = int(data.get("LTV_Ratio", 70))

  tot_inc = max(app_inc + co_inc, 1.0)
  new_emi = calculate_emi(loan_amt, rate, tenure)
  foir = ((existing_emi + new_emi) / tot_inc) * 100.0
  total_payable = new_emi * tenure
  total_interest = max(total_payable - loan_amt, 0.0)

  # Max Eligible Loan at safe 50% FOIR cap
  max_emi_allowed = max((0.50 * tot_inc) - existing_emi, 0.0)
  if max_emi_allowed > 0 and tenure > 0:
    r = rate / (12.0 * 100.0)
    factor = math.pow(1 + r, tenure) if r > 0 else 1
    max_eligible_loan = (
        max_emi_allowed * (factor - 1) / (r * factor)
        if r > 0
        else max_emi_allowed * tenure
    )
  else:
    max_eligible_loan = 0.0

  # Find Optimal Tenure if FOIR > 50%
  suggested_tenure = None
  if foir > 50.0:
    for candidate_t in [120, 180, 240, 300, 360]:
      if candidate_t > tenure:
        cand_emi = calculate_emi(loan_amt, rate, candidate_t)
        cand_foir = ((existing_emi + cand_emi) / tot_inc) * 100.0
        if cand_foir <= 50.0:
          suggested_tenure = candidate_t
          break

  sample = np.array(
      [[
          app_inc,
          co_inc,
          existing_emi,
          loan_amt,
          tenure,
          rate,
          cibil,
          emp_type,
          years_job,
          prop_area,
          ltv_ratio,
          round(foir, 2),
      ]],
      dtype=float,
  )
  sample_s = (sample - mean_vec) / std_vec

  prob_rf = float(rf_model.predict_proba(sample_s)[0])
  prob_gb = float(gbdt_model.predict_proba(sample_s)[0])
  prob_lr = float(lr_model.predict_proba(sample_s)[0])
  prob_ens = 0.50 * prob_rf + 0.35 * prob_gb + 0.15 * prob_lr

  # Internal Risk Score (0-1000) & Credit Rating
  cibil_norm = np.clip((cibil - 500) / 400.0, 0, 1) * 450
  foir_norm = np.clip((75.0 - foir) / 60.0, 0, 1) * 350
  ltv_norm = np.clip((100.0 - ltv_ratio) / 55.0, 0, 1) * 120
  exp_norm = np.clip(years_job / 10.0, 0, 1) * 80
  risk_score = int(round(cibil_norm + foir_norm + ltv_norm + exp_norm))

  if risk_score >= 820:
    grade, rec_rate = "AAA (Prime)", 8.45
  elif risk_score >= 730:
    grade, rec_rate = "AA (Low Risk)", 8.95
  elif risk_score >= 640:
    grade, rec_rate = "A (Standard)", 9.60
  elif risk_score >= 540:
    grade, rec_rate = "BBB (Moderate)", 10.85
  else:
    grade, rec_rate = "C (Sub-Prime)", 13.50

  # XAI Audit Drivers & Policy Guardrails
  reasons = []
  hard_reject = False

  if cibil >= 750:
    reasons.append({
        "type": "positive",
        "impact": "+38%",
        "text": f"Prime CIBIL Score ({cibil}) strongly lowers default probability.",
    })
  elif cibil >= 680:
    reasons.append({
        "type": "positive",
        "impact": "+16%",
        "text": f"Acceptable CIBIL Score ({cibil}) satisfies minimum bureau cutoff.",
    })
  else:
    reasons.append({
        "type": "negative",
        "impact": "-52%",
        "text": f"Sub-prime CIBIL ({cibil}) violates the 680 minimum credit policy.",
    })
    hard_reject = True

  if foir <= 45.0:
    reasons.append({
        "type": "positive",
        "impact": "+32%",
        "text": f"Strong repayment capacity with FOIR at {foir:.1f}% (<= 45% ideal).",
    })
  elif foir <= 55.0:
    reasons.append({
        "type": "warning",
        "impact": "-8%",
        "text": f"Elevated FOIR ({foir:.1f}%) approaches the 55% regulatory ceiling.",
    })
  else:
    reasons.append({
        "type": "negative",
        "impact": "-48%",
        "text": f"Excessive Debt Burden: FOIR ({foir:.1f}%) breaches the 55% maximum cap.",
    })
    hard_reject = True

  if ltv_ratio <= 80:
    reasons.append({
        "type": "positive",
        "impact": "+14%",
        "text": f"Healthy Loan-to-Value collateral coverage ({ltv_ratio}% LTV).",
    })
  elif ltv_ratio <= 85:
    reasons.append({
        "type": "warning",
        "impact": "-6%",
        "text": f"High LTV Ratio ({ltv_ratio}%) reduces collateral equity buffer.",
    })
  else:
    reasons.append({
        "type": "negative",
        "impact": "-28%",
        "text": f"LTV Ratio ({ltv_ratio}%) exceeds 85% maximum collateral limit.",
    })
    hard_reject = True

  if years_job < 1 and emp_type != 0:
    reasons.append({
        "type": "negative",
        "impact": "-22%",
        "text": "Insufficient employment stability (< 1 year in current role/business).",
    })
    hard_reject = True

  if hard_reject:
    status = "REJECTED"
    confidence = max(1.0 - prob_ens, 0.93)
  else:
    status = "APPROVED" if prob_ens >= 0.5 else "REJECTED"
    confidence = prob_ens if status == "APPROVED" else (1.0 - prob_ens)

  # Counter-Offer Suggestions
  counter_offers = []
  if status == "REJECTED":
    if max_eligible_loan >= 100000 and loan_amt > max_eligible_loan:
      counter_offers.append(
          f"Reduce requested principal to â‚¹{max_eligible_loan:,.0f} to keep FOIR at 50%."
      )
    if suggested_tenure:
      counter_offers.append(
          f"Extend loan tenure to {suggested_tenure} months to bring monthly EMI within safe limits."
      )
    if cibil < 680:
      counter_offers.append(
          "Improve CIBIL score above 680 or add a co-applicant with strong credit history."
      )
    if ltv_ratio > 85:
      counter_offers.append(
          "Increase down payment to reduce Loan-to-Value (LTV) ratio to 80% or lower."
      )

  amortization = generate_amortization(loan_amt, rate, tenure)

  print(
      f"[UNDERWRITE] Grade: {grade.split()[0]} | Score: {risk_score} | FOIR:"
      f" {foir:.1f}% | Ensemble: {prob_ens*100:.1f}% ==> {status}"
  )

  return jsonify({
      "status": status,
      "confidence": round(confidence * 100, 1),
      "risk_score": risk_score,
      "credit_grade": grade,
      "recommended_rate": rec_rate,
      "monthly_emi": round(new_emi, 0),
      "total_interest": round(total_interest, 0),
      "total_payable": round(total_payable, 0),
      "foir": round(foir, 1),
      "max_eligible_loan": round(max_eligible_loan, 0),
      "model_votes": {
          "ensemble": round(prob_ens * 100, 1),
          "random_forest": round(prob_rf * 100, 1),
          "gradient_boosting": round(prob_gb * 100, 1),
          "logistic_regression": round(prob_lr * 100, 1),
      },
      "benchmarks": MODEL_BENCHMARKS,
      "feature_importances": FEATURE_IMPORTANCES,
      "reasons": reasons,
      "counter_offers": counter_offers,
      "amortization": amortization,
  })


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
