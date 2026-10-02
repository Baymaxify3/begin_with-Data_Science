"""
Student Exam Performance - ML Pipeline
Targets:
  1. Pass/Fail classification
  2. Grade (A/B/C/D/F) classification
  3. Exam score regression
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, r2_score
)

# ── 1. LOAD DATA (sample 10,000 rows for speed) ─────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(HERE, "student_exam_performance.csv")

print("=" * 55)
print("  STUDENT PERFORMANCE - ML PIPELINE")
print("=" * 55)

df = pd.read_csv(CSV)
print("Loaded:", len(df), "rows x", len(df.columns), "columns")

print("\nPass/Fail counts:")
print(df["pass_status"].value_counts().to_string())
print("\nGrade counts:")
print(df["performance_grade"].value_counts().sort_index().to_string())
print("\nExam Score stats:")
print(df["exam_score"].describe().round(2).to_string())

# ── 2. FEATURES ──────────────────────────────────────────────────────────────
NUM_COLS = [
    "age", "previous_exam_score", "previous_gpa",
    "attendance_percentage", "assignment_completion_rate",
    "study_hours_per_day", "self_study_hours", "online_learning_hours",
    "practice_tests_completed", "sleep_hours", "daily_screen_time",
    "physical_activity_hours", "online_course_hours",
    "exam_preparation_days", "time_management_score", "exam_anxiety_level",
    "stress_level"
]

CAT_COLS = [
    "gender", "education_level", "school_type", "family_income",
    "parent_education", "urban_rural", "class_participation",
    "study_consistency", "study_environment", "study_method",
    "revision_frequency", "notes_quality", "sleep_quality",
    "break_frequency", "motivation_level", "device_availability",
    "educational_app_usage", "exam_difficulty"
]

BIN_COLS = ["private_tuition", "internet_access"]

# Keep only columns present in the dataset
NUM_COLS = [c for c in NUM_COLS if c in df.columns]
CAT_COLS = [c for c in CAT_COLS if c in df.columns]
BIN_COLS = [c for c in BIN_COLS if c in df.columns]

ALL_COLS = NUM_COLS + CAT_COLS + BIN_COLS
X = df[ALL_COLS].copy()
print("\nFeatures used:", len(ALL_COLS),
      " (numeric:", len(NUM_COLS), " cat:", len(CAT_COLS), " binary:", len(BIN_COLS), ")")

# ── 3. PREPROCESSOR FACTORY ──────────────────────────────────────────────────
def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler())
        ]), NUM_COLS),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), CAT_COLS),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN_COLS),
    ])

# ── 4. TASK 1 — PASS / FAIL CLASSIFICATION ───────────────────────────────────
print("\n" + "=" * 55)
print("  TASK 1: PASS / FAIL  (Binary Classification)")
print("=" * 55)

y_pass = df["pass_status"].map({"Pass": 1, "Fail": 0})
Xtr, Xte, ytr, yte = train_test_split(X, y_pass, test_size=0.2,
                                       random_state=42, stratify=y_pass)

pf_models = {
    "Logistic Regression": Pipeline([("pre", make_pre()),
        ("clf", LogisticRegression(max_iter=500, random_state=42))]),
    "Random Forest":       Pipeline([("pre", make_pre()),
        ("clf", RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=42))]),
}

best_pf_name, best_pf_model, best_pf_acc = "", None, 0
for name, mdl in pf_models.items():
    mdl.fit(Xtr, ytr)
    preds = mdl.predict(Xte)
    acc   = accuracy_score(yte, preds)
    print(f"\n{name}  ->  Accuracy: {acc*100:.1f}%")
    print(classification_report(yte, preds, target_names=["Fail","Pass"]))
    if acc > best_pf_acc:
        best_pf_acc, best_pf_name, best_pf_model = acc, name, mdl

print(f"[BEST] Pass/Fail: {best_pf_name}  ({best_pf_acc*100:.1f}%)")

# ── 5. TASK 2 — GRADE CLASSIFICATION ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  TASK 2: GRADE A/B/C/D/F  (Multi-class)")
print("=" * 55)

y_grade = df["performance_grade"]
Xtr_g, Xte_g, ytr_g, yte_g = train_test_split(X, y_grade, test_size=0.2,
                                                random_state=42, stratify=y_grade)

gr_model = Pipeline([("pre", make_pre()),
    ("clf", RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=42))])
gr_model.fit(Xtr_g, ytr_g)
preds_g = gr_model.predict(Xte_g)
gr_acc  = accuracy_score(yte_g, preds_g)
print(f"Random Forest  ->  Accuracy: {gr_acc*100:.1f}%")
print(classification_report(yte_g, preds_g))

# ── 6. TASK 3 — EXAM SCORE REGRESSION ────────────────────────────────────────
print("\n" + "=" * 55)
print("  TASK 3: EXAM SCORE  (Regression)")
print("=" * 55)

y_score = df["exam_score"]
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y_score, test_size=0.2,
                                                random_state=42)

reg_models = {
    "Linear Regression":      Pipeline([("pre", make_pre()), ("reg", LinearRegression())]),
    "Random Forest Regressor":Pipeline([("pre", make_pre()),
        ("reg", RandomForestRegressor(n_estimators=30, n_jobs=-1, random_state=42))]),
}

best_reg_name, best_reg_model, best_r2 = "", None, -999
for name, mdl in reg_models.items():
    mdl.fit(Xtr_r, ytr_r)
    preds = mdl.predict(Xte_r)
    mae   = mean_absolute_error(yte_r, preds)
    r2    = r2_score(yte_r, preds)
    print(f"{name}  ->  MAE: {mae:.2f}  R2: {r2:.4f}")
    if r2 > best_r2:
        best_r2, best_reg_name, best_reg_model = r2, name, mdl

print(f"[BEST] Regression: {best_reg_name}  (R2={best_r2:.4f})")

# ── 7. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  TOP 15 FEATURE IMPORTANCES  (Pass/Fail Random Forest)")
print("=" * 55)

rf_model = pf_models["Random Forest"]
ohe_names = list(rf_model.named_steps["pre"]
                 .named_transformers_["cat"]
                 .named_steps["ohe"]
                 .get_feature_names_out(CAT_COLS))
feat_names = NUM_COLS + ohe_names + BIN_COLS
importances = pd.Series(rf_model.named_steps["clf"].feature_importances_,
                        index=feat_names).nlargest(15).sort_values()
print(importances.to_string())

# ── 8. CHARTS ────────────────────────────────────────────────────────────────
print("\nGenerating charts ...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Student Performance - ML Results", fontsize=15, fontweight="bold")

# (a) Confusion matrix - Pass/Fail
ax = axes[0, 0]
cm = confusion_matrix(yte, best_pf_model.predict(Xte))
ConfusionMatrixDisplay(cm, display_labels=["Fail","Pass"]).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"Pass/Fail Confusion Matrix\n({best_pf_name}, {best_pf_acc*100:.1f}%)")

# (b) Confusion matrix - Grade
ax = axes[0, 1]
cm_g = confusion_matrix(yte_g, preds_g, labels=["A","B","C","D","F"])
ConfusionMatrixDisplay(cm_g, display_labels=["A","B","C","D","F"]).plot(ax=ax, colorbar=False, cmap="Purples")
ax.set_title(f"Grade Confusion Matrix\n(Random Forest, {gr_acc*100:.1f}%)")

# (c) Actual vs Predicted - Score
ax = axes[0, 2]
score_preds = best_reg_model.predict(Xte_r)
ax.scatter(yte_r, score_preds, alpha=0.2, s=6, color="#3b82d4")
mn, mx = float(yte_r.min()), float(yte_r.max())
ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect")
ax.set_xlabel("Actual Score"); ax.set_ylabel("Predicted Score")
ax.set_title(f"Actual vs Predicted Score\n({best_reg_name}, R2={best_r2:.3f})")
ax.legend(fontsize=8)

# (d) Feature importance
ax = axes[1, 0]
colors = ["#3b82d4" if i >= 10 else "#94a3b8" for i in range(len(importances))]
importances.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Top 15 Feature Importances\n(Random Forest - Pass/Fail)")
ax.set_xlabel("Importance")
ax.tick_params(axis="y", labelsize=8)

# (e) Grade actual vs predicted bar
ax = axes[1, 1]
order = ["A","B","C","D","F"]
actual_c = pd.Series(yte_g).value_counts().reindex(order).fillna(0)
pred_c   = pd.Series(preds_g).value_counts().reindex(order).fillna(0)
x = np.arange(5); w = 0.35
ax.bar(x - w/2, actual_c, w, label="Actual",    color="#3b82d4", alpha=0.85)
ax.bar(x + w/2, pred_c,   w, label="Predicted", color="#7c5cd8", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(order)
ax.set_title("Grade Distribution\nActual vs Predicted")
ax.set_xlabel("Grade"); ax.set_ylabel("Count"); ax.legend(fontsize=8)

# (f) Residuals
ax = axes[1, 2]
residuals = np.array(yte_r) - score_preds
ax.hist(residuals, bins=40, color="#22c55e", edgecolor="white", alpha=0.85)
ax.axvline(0, color="red", lw=1.5, linestyle="--")
ax.set_title(f"Score Residuals\n(mean error: {residuals.mean():.2f})")
ax.set_xlabel("Residual (Actual - Predicted)"); ax.set_ylabel("Count")

plt.tight_layout()
out = os.path.join(HERE, "student_ml_results.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
plt.close()
print("[OK] Chart saved ->", out)

# ── 9. LIVE PREDICTION EXAMPLE ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  LIVE PREDICTION EXAMPLE")
print("=" * 55)

sample = pd.DataFrame([{
    "age": 17, "previous_exam_score": 65.0, "previous_gpa": 2.8,
    "attendance_percentage": 88.0, "assignment_completion_rate": 72.0,
    "study_hours_per_day": 3.5, "self_study_hours": 2.0,
    "online_learning_hours": 1.5, "practice_tests_completed": 5,
    "sleep_hours": 7.0, "daily_screen_time": 4.0,
    "physical_activity_hours": 1.0, "online_course_hours": 2.0,
    "exam_preparation_days": 14, "time_management_score": 7.0,
    "exam_anxiety_level": 6.0, "stress_level": 7,
    "gender": "Female", "education_level": "High School",
    "school_type": "Public", "family_income": "Middle",
    "parent_education": "Bachelor", "urban_rural": "Suburban",
    "class_participation": "Medium", "study_consistency": "Medium",
    "study_environment": "Quiet", "study_method": "Flashcards",
    "revision_frequency": "Weekly", "notes_quality": "Average",
    "sleep_quality": "Good", "break_frequency": "Occasionally",
    "motivation_level": "Medium", "device_availability": "Dedicated",
    "educational_app_usage": "Moderate", "exam_difficulty": "Medium",
    "private_tuition": 0, "internet_access": 1,
}])

pf_pred   = best_pf_model.predict(sample)[0]
pf_prob   = best_pf_model.predict_proba(sample)[0]
gr_pred   = gr_model.predict(sample)[0]
sc_pred   = best_reg_model.predict(sample)[0]

print("Profile: 17yr Female | High School | 3.5 hrs study/day | Medium stress")
print("Pass/Fail  :", "PASS" if pf_pred == 1 else "FAIL",
      f"  (confidence: {max(pf_prob)*100:.1f}%)")
print("Grade      :", gr_pred)
print("Exam Score :", round(sc_pred, 1), "/ 100")

print("\n" + "=" * 55)
print("  PIPELINE COMPLETE")
print("=" * 55)
