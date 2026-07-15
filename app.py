"""
Cardio Guard - Heart Disease Prediction
Model Training Script  (v4 — Ensemble Learning + XAI)
Dataset: heart_statlog_cleveland_hungary_final.csv

NEW:
  • Ensemble: VotingClassifier (soft voting) + StackingClassifier
  • XAI:      SHAP values saved as a static chart (shap_summary.png)
  • HTML:     All EDA charts also exported as a standalone HTML report
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import shap

os.makedirs("data",          exist_ok=True)
os.makedirs("models",        exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# ── 1. LOAD & PREPROCESS ─────────────────────────────────────
df = pd.read_csv("data/heart_statlog_cleveland_hungary_final.csv")
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df["target"] = (df["target"] > 0).astype(int)
print(f"✅ Loaded {df.shape[0]} rows | Features: {df.columns.drop('target').tolist()}")

X = df.drop("target", axis=1)
y = df["target"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ── 2. EDA PLOTS ─────────────────────────────────────────────
def save(fname): plt.tight_layout(); plt.savefig(f"static/images/{fname}", dpi=120); plt.close()

# Target distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["target"].value_counts()
ax.bar(["No Disease", "Heart Disease"], counts.values, color=["#26c6da", "#ef5350"])
ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold"); ax.set_ylabel("Count")
for i, v in enumerate(counts.values): ax.text(i, v+5, str(v), ha="center", fontweight="bold")
save("target_dist.png")

# Age distribution
fig, ax = plt.subplots(figsize=(7, 4))
for label, color, name in [(0,"#26c6da","No Disease"),(1,"#ef5350","Heart Disease")]:
    df[df["target"]==label]["age"].hist(bins=20, alpha=0.7, color=color, label=name, ax=ax)
ax.set_title("Age Distribution by Heart Disease", fontsize=14, fontweight="bold")
ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.legend()
save("age_distribution.png")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, mask=np.triu(np.ones_like(corr,dtype=bool)), annot=True,
            fmt=".2f", cmap="RdYlGn", center=0, ax=ax, linewidths=0.5, annot_kws={"size":7})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
save("correlation_heatmap.png")
print("✅ EDA plots saved")

# ── 3. BASE MODELS ───────────────────────────────────────────
base_clfs = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=7),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, random_state=42),
}

scaler  = StandardScaler()
results = {}

for name, clf in base_clfs.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {"pipeline": pipe, "accuracy": acc, "auc": auc,
                     "cv_mean": cv.mean(), "y_pred": y_pred, "y_proba": y_proba}
    print(f"  {name:25s}  Acc={acc:.4f}  AUC={auc:.4f}  CV={cv.mean():.4f}")

# ── 4. ENSEMBLE MODELS ───────────────────────────────────────
print("\n🔗 Training Ensemble Models …")

# Scale once for ensemble (shared scaler)
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 4a. Soft Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ("lr",  LogisticRegression(max_iter=1000, random_state=42)),
        ("rf",  RandomForestClassifier(n_estimators=200, random_state=42)),
        ("gb",  GradientBoostingClassifier(n_estimators=150, random_state=42)),
        ("svm", SVC(probability=True, random_state=42)),
    ],
    voting="soft",
)
voting_clf.fit(X_train_sc, y_train)
vot_pred  = voting_clf.predict(X_test_sc)
vot_proba = voting_clf.predict_proba(X_test_sc)[:, 1]
vot_acc   = accuracy_score(y_test, vot_pred)
vot_auc   = roc_auc_score(y_test, vot_proba)
cv_vot    = cross_val_score(voting_clf, X_train_sc, y_train, cv=5, scoring="accuracy")
results["Voting Ensemble"] = {"pipeline": None, "accuracy": vot_acc, "auc": vot_auc,
                               "cv_mean": cv_vot.mean(), "y_pred": vot_pred, "y_proba": vot_proba}
print(f"  {'Voting Ensemble':25s}  Acc={vot_acc:.4f}  AUC={vot_auc:.4f}  CV={cv_vot.mean():.4f}")

# 4b. Stacking Classifier (meta-learner = Logistic Regression)
stacking_clf = StackingClassifier(
    estimators=[
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42)),
        ("gb",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ("svm", SVC(probability=True, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=7)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    passthrough=False,
)
stacking_clf.fit(X_train_sc, y_train)
stk_pred  = stacking_clf.predict(X_test_sc)
stk_proba = stacking_clf.predict_proba(X_test_sc)[:, 1]
stk_acc   = accuracy_score(y_test, stk_pred)
stk_auc   = roc_auc_score(y_test, stk_proba)
cv_stk    = cross_val_score(stacking_clf, X_train_sc, y_train, cv=5, scoring="accuracy")
results["Stacking Ensemble"] = {"pipeline": None, "accuracy": stk_acc, "auc": stk_auc,
                                 "cv_mean": cv_stk.mean(), "y_pred": stk_pred, "y_proba": stk_proba}
print(f"  {'Stacking Ensemble':25s}  Acc={stk_acc:.4f}  AUC={stk_auc:.4f}  CV={cv_stk.mean():.4f}")

# ── 5. CHARTS ────────────────────────────────────────────────
names  = list(results.keys())
colors = ["#ef5350","#26c6da","#66bb6a","#ffa726","#ab47bc","#26a69a","#ff7043","#5c6bc0"]

# Model comparison (now includes ensemble)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
accs = [results[n]["accuracy"] for n in names]
aucs = [results[n]["auc"]      for n in names]
clrs = colors[:len(names)]
axes[0].barh(names, accs, color=clrs); axes[0].set_xlim(0.5, 1.0)
axes[0].set_title("Accuracy Comparison — All Models incl. Ensemble", fontweight="bold")
axes[0].set_xlabel("Accuracy")
for i, v in enumerate(accs): axes[0].text(v+.002, i, f"{v:.3f}", va="center")
axes[1].barh(names, aucs, color=clrs); axes[1].set_xlim(0.5, 1.0)
axes[1].set_title("AUC-ROC Comparison — All Models incl. Ensemble", fontweight="bold")
axes[1].set_xlabel("AUC")
for i, v in enumerate(aucs): axes[1].text(v+.002, i, f"{v:.3f}", va="center")
plt.tight_layout(); plt.savefig("static/images/model_comparison.png", dpi=120); plt.close()
# Best model (by AUC)
best_name = max(results, key=lambda n: results[n]["auc"])
best      = results[best_name]
print(f"\n🏆 Best: {best_name}  AUC={best['auc']:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, best["y_pred"])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Disease","Heart Disease"],
            yticklabels=["No Disease","Heart Disease"])
ax.set_title(f"Confusion Matrix – {best_name}", fontweight="bold")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
save("confusion_matrix.png")

# ROC curves
fig, ax = plt.subplots(figsize=(8, 5))
for name, color in zip(names, clrs):
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
    ls = "--" if "Ensemble" in name else "-"
    lw = 2.5 if "Ensemble" in name else 1.5
    ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
            label=f"{name} (AUC={results[name]['auc']:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_title("ROC Curves – All Models (dashed = Ensemble)", fontweight="bold")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=7.5); save("roc_curves.png")

# Feature importance (RF)
rf_pipe = results["Random Forest"]["pipeline"]
imps    = rf_pipe.named_steps["clf"].feature_importances_
idx     = np.argsort(imps)[::-1]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(len(feature_names)), imps[idx], color="#26c6da")
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels([feature_names[i] for i in idx], rotation=40, ha="right")
ax.set_title("Feature Importances – Random Forest", fontweight="bold")
ax.set_ylabel("Importance"); save("feature_importance.png")

# ── 6. XAI — SHAP VALUES ─────────────────────────────────────
print("\n🔍 Computing SHAP values …")
try:
    # Use the RF pipeline's scaler + RF model directly
    rf_model  = rf_pipe.named_steps["clf"]
    X_test_sc_rf = rf_pipe.named_steps["scaler"].transform(X_test)
    explainer  = shap.TreeExplainer(rf_model)
    shap_vals  = explainer.shap_values(X_test_sc_rf)

    # For binary classification shap_values returns list [class0, class1]
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    fig, ax = plt.subplots(figsize=(9, 5))
    shap.summary_plot(sv, X_test_sc_rf, feature_names=feature_names,
                      plot_type="bar", show=False, color="#ef5350")
    plt.title("SHAP Feature Importance (Random Forest)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("static/images/shap_summary.png", dpi=120)
    plt.close()
    print("✅ SHAP summary saved → static/images/shap_summary.png")

    # Beeswarm plot
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.summary_plot(sv, X_test_sc_rf, feature_names=feature_names, show=False)
    plt.title("SHAP Beeswarm Plot – Impact of each feature", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("static/images/shap_beeswarm.png", dpi=120)
    plt.close()
    print("✅ SHAP beeswarm saved → static/images/shap_beeswarm.png")

    joblib.dump(explainer, "models/shap_explainer.pkl")
    joblib.dump(rf_pipe.named_steps["scaler"], "models/scaler.pkl")
    print("✅ SHAP explainer saved")
except Exception as e:
    print(f"⚠️  SHAP skipped: {e}")

# ── 7. HTML REPORT ───────────────────────────────────────────
print("\n📄 Generating HTML report …")

chart_list = [
    ("Target Distribution",        "target_dist.png"),
    ("Age Distribution",           "age_distribution.png"),
    ("Correlation Heatmap",        "correlation_heatmap.png"),
    ("Model Comparison",           "model_comparison.png"),
    ("Confusion Matrix",           "confusion_matrix.png"),
    ("ROC Curves",                 "roc_curves.png"),
    ("Feature Importances (RF)",   "feature_importance.png"),
    ("SHAP Importance",            "shap_summary.png"),
    ("SHAP Beeswarm",              "shap_beeswarm.png"),
]

# Build metrics table rows
rows_html = ""
for n in names:
    r = results[n]
    tag = " 🏆" if n == best_name else ""
    ens = " <span class='ens-badge'>Ensemble</span>" if "Ensemble" in n else ""
    rows_html += (
        f"<tr><td>{n}{ens}{tag}</td>"
        f"<td>{r['accuracy']:.4f}</td>"
        f"<td>{r['auc']:.4f}</td>"
        f"<td>{r['cv_mean']:.4f}</td></tr>\n"
    )

# Build chart cards
charts_html = ""
for title, fname in chart_list:
    path = f"static/images/{fname}"
    if os.path.exists(path):
        charts_html += f"""
        <div class="card">
          <h3>{title}</h3>
          <img src="{path}" alt="{title}">
        </div>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cardio Guard – Model Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #f4f6f9; color: #212121; }}
  header {{
    background: linear-gradient(135deg, #b71c1c, #ef5350 55%, #26c6da);
    color: #fff; padding: 2rem 3rem;
  }}
  header h1 {{ font-size: 2rem; font-weight: 700; }}
  header p  {{ opacity: .88; margin-top: .4rem; }}
  .container {{ max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; }}
  h2 {{ font-size: 1.3rem; font-weight: 700; margin: 2rem 0 1rem;
        padding-bottom: .4rem; border-bottom: 3px solid #ef5350; color: #b71c1c; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff;
           border-radius: 10px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,.07); }}
  th {{ background: #b71c1c; color: #fff; padding: .75rem 1rem; text-align: left; font-size: .85rem; }}
  td {{ padding: .7rem 1rem; border-bottom: 1px solid #f0f0f0; font-size: .9rem; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #fff5f5; }}
  .ens-badge {{ background: #ef9a9a; color: #b71c1c; font-size: .7rem; font-weight: 700;
                padding: 1px 6px; border-radius: 4px; margin-left: 5px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 1.5rem; margin-top: 1rem; }}
  .card {{ background: #fff; border-radius: 12px; padding: 1.25rem;
           box-shadow: 0 2px 12px rgba(0,0,0,.07); }}
  .card h3 {{ font-size: .95rem; font-weight: 600; margin-bottom: .75rem; color: #444; }}
  .card img {{ width: 100%; border-radius: 6px; border: 1px solid #eee; }}
  footer {{ text-align: center; padding: 2rem; font-size: .8rem; color: #9e9e9e; }}
</style>
</head>
<body>
<header>
  <h1>❤️ Cardio Guard — Model Report</h1>
  <p>Heart Disease Prediction · Ensemble Learning + XAI · BSCS ML Project 2025</p>
</header>
<div class="container">

  <h2>Model Performance Summary</h2>
  <table>
    <thead><tr><th>Model</th><th>Accuracy</th><th>AUC-ROC</th><th>CV Score (5-Fold)</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>

  <h2>Visualisations & XAI Charts</h2>
  <div class="grid">
    {charts_html}
  </div>

</div>
<footer>Cardio Guard · Dataset: heart_statlog_cleveland_hungary_final.csv · 1,190 samples · 11 features</footer>
</body>
</html>"""

with open("static/report.html", "w", encoding="utf-8") as f:
    f.write(html)
print("✅ HTML report saved → static/report.html")

# ── 8. SAVE BEST MODEL ───────────────────────────────────────
# Best model: if it's an ensemble use the scaler + ensemble dict
if best_name in ("Voting Ensemble", "Stacking Ensemble"):
    clf_obj = voting_clf if best_name == "Voting Ensemble" else stacking_clf
    joblib.dump({"type": "ensemble", "clf": clf_obj, "scaler": scaler},
                "models/best_model.pkl")
else:
    joblib.dump(results[best_name]["pipeline"], "models/best_model.pkl")

joblib.dump(feature_names, "models/feature_names.pkl")
with open("models/best_model_name.txt", "w") as f:
    f.write(best_name)
# Save scaler always (needed for SHAP in app)
joblib.dump(scaler, "models/scaler.pkl")

print(f"\n💾 Best model saved: {best_name}")
print(classification_report(y_test, best["y_pred"], target_names=["No Disease","Heart Disease"]))
print("✅ Done!")
