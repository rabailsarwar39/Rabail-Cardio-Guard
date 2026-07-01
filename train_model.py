"""
Cardio Guard - Heart Disease Prediction
Model Training Script
Dataset: heart_statlog_cleveland_hungary_final.csv
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

# ── 1. LOAD ──────────────────────────────────────────────────
df = pd.read_csv("data/heart_statlog_cleveland_hungary_final.csv")
print("✅ Loaded:", df.shape)
print("Columns:", df.columns.tolist())

# Normalise column names: lowercase + underscores
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print("Normalised:", df.columns.tolist())

# Binary target
df["target"] = (df["target"] > 0).astype(int)
print("Target distribution:\n", df["target"].value_counts())

# ── 2. EDA PLOTS ─────────────────────────────────────────────
# Target distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df["target"].value_counts()
ax.bar(["No Disease", "Heart Disease"], counts.values, color=["#26c6da", "#ef5350"])
ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold")
ax.set_ylabel("Count")
for i, v in enumerate(counts.values):
    ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("static/images/target_dist.png", dpi=120); plt.close()

# Age distribution
fig, ax = plt.subplots(figsize=(7, 4))
for label, color, name in [(0, "#26c6da", "No Disease"), (1, "#ef5350", "Heart Disease")]:
    df[df["target"] == label]["age"].hist(bins=20, alpha=0.7, color=color, label=name, ax=ax)
ax.set_title("Age Distribution by Heart Disease", fontsize=14, fontweight="bold")
ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.legend()
plt.tight_layout()
plt.savefig("static/images/age_distribution.png", dpi=120); plt.close()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, ax=ax, linewidths=0.5, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("static/images/correlation_heatmap.png", dpi=120); plt.close()

print("✅ EDA plots saved")

# ── 3. TRAIN ─────────────────────────────────────────────────
X = df.drop("target", axis=1)
y = df["target"]
feature_names = X.columns.tolist()
print("Features:", feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, random_state=42),
    "SVM":                  SVC(probability=True, random_state=42),
    "KNN":                  KNeighborsClassifier(n_neighbors=7),
    "Decision Tree":        DecisionTreeClassifier(max_depth=5, random_state=42),
}

results = {}
for name, clf in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {"pipeline": pipe, "accuracy": acc, "auc": auc,
                     "cv_mean": cv.mean(), "y_pred": y_pred, "y_proba": y_proba}
    print(f"  {name:25s} Acc={acc:.4f}  AUC={auc:.4f}  CV={cv.mean():.4f}")

# Model comparison plot
names  = list(results.keys())
colors = ["#ef5350","#26c6da","#66bb6a","#ffa726","#ab47bc","#26a69a"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
accs = [results[n]["accuracy"] for n in names]
aucs = [results[n]["auc"]      for n in names]
axes[0].barh(names, accs, color=colors); axes[0].set_xlim(0.5, 1.0)
axes[0].set_title("Accuracy Comparison", fontweight="bold"); axes[0].set_xlabel("Accuracy")
for i, v in enumerate(accs): axes[0].text(v+.002, i, f"{v:.3f}", va="center")
axes[1].barh(names, aucs, color=colors); axes[1].set_xlim(0.5, 1.0)
axes[1].set_title("AUC-ROC Comparison", fontweight="bold"); axes[1].set_xlabel("AUC")
for i, v in enumerate(aucs): axes[1].text(v+.002, i, f"{v:.3f}", va="center")
plt.tight_layout()
plt.savefig("static/images/model_comparison.png", dpi=120); plt.close()

# Best model
best_name = max(results, key=lambda n: results[n]["auc"])
best = results[best_name]
print(f"\n🏆 Best: {best_name}  AUC={best['auc']:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, best["y_pred"])
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No Disease","Heart Disease"],
            yticklabels=["No Disease","Heart Disease"])
ax.set_title(f"Confusion Matrix – {best_name}", fontweight="bold")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("static/images/confusion_matrix.png", dpi=120); plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(7, 5))
for name, color in zip(names, colors):
    fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
    ax.plot(fpr, tpr, color=color, lw=1.5,
            label=f"{name} (AUC={results[name]['auc']:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_title("ROC Curves – All Models", fontweight="bold")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("static/images/roc_curves.png", dpi=120); plt.close()

# Feature importance
rf_pipe = results["Random Forest"]["pipeline"]
importances = rf_pipe.named_steps["clf"].feature_importances_
idx = np.argsort(importances)[::-1]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(range(len(feature_names)), importances[idx], color="#26c6da")
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels([feature_names[i] for i in idx], rotation=40, ha="right")
ax.set_title("Feature Importances – Random Forest", fontweight="bold")
ax.set_ylabel("Importance")
plt.tight_layout()
plt.savefig("static/images/feature_importance.png", dpi=120); plt.close()

print("✅ All plots saved")
print(classification_report(y_test, best["y_pred"], target_names=["No Disease","Heart Disease"]))

# ── 4. SAVE ──────────────────────────────────────────────────
joblib.dump(best["pipeline"], "models/best_model.pkl")
joblib.dump(feature_names,    "models/feature_names.pkl")
with open("models/best_model_name.txt", "w") as f:
    f.write(best_name)
print("💾 Model saved → models/best_model.pkl")
print("✅ Done!")
