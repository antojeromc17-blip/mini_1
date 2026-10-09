"""
=============================================================================
PHISHING WEBSITE DETECTION - Model Training & Evaluation
KTU S3 CSE AI & ML Mini Project
=============================================================================
This script:
  1. Downloads the UCI Phishing Websites dataset
  2. Performs Exploratory Data Analysis (EDA)
  3. Trains 4 ML models (Logistic Regression, Decision Tree, Random Forest, KNN)
  4. Evaluates & compares all models
  5. Saves the best model + scaler for the Streamlit app
  6. Generates all charts/plots in the 'outputs/' folder
=============================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -- Create output directories -----------------------------------------------
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('data', exist_ok=True)

# =============================================================================
# STEP 1 - LOAD DATASET
# =============================================================================
print("=" * 60)
print("STEP 1: Loading UCI Phishing Websites Dataset...")
print("=" * 60)

csv_path = 'data/phishing_dataset.csv'
if os.path.exists(csv_path):
    print(f"[INFO] Found local dataset at {csv_path}, loading...")
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['Result'])
    y = df['Result'].values.ravel()
else:
    from ucimlrepo import fetch_ucirepo
    phishing_websites = fetch_ucirepo(id=327)
    X = phishing_websites.data.features
    y = phishing_websites.data.targets
    y = y.values.ravel()
    df = X.copy()
    df['Result'] = y
    df.to_csv(csv_path, index=False)

print("[OK] Dataset loaded successfully!")
print(f"   Shape: {X.shape[0]} rows x {X.shape[1]} features")
print(f"   Target classes: {np.unique(y)}")
print(f"   Saved to {csv_path}")

# =============================================================================
# STEP 2 - EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

# --- 2a. Basic Info ---
print(f"\n[INFO] Dataset Overview:")
print(f"   Total samples: {df.shape[0]}")
print(f"   Total features: {X.shape[1]}")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Data types:\n{X.dtypes.value_counts().to_string()}")

# --- 2b. Class Distribution ---
print(f"\n[INFO] Class Distribution:")
class_counts = pd.Series(y).value_counts()
for cls, count in class_counts.items():
    label = "Legitimate" if cls == 1 else "Phishing"
    pct = count / len(y) * 100
    print(f"   {label} ({cls}): {count} ({pct:.1f}%)")

# Plot class distribution
plt.figure(figsize=(8, 5))
colors = ['#e74c3c', '#2ecc71']  # Red for phishing, Green for legitimate
labels_map = {-1: 'Phishing (-1)', 1: 'Legitimate (1)'}
bars = plt.bar(
    [labels_map[c] for c in class_counts.index],
    class_counts.values,
    color=colors,
    edgecolor='black',
    linewidth=0.8
)
for bar, count in zip(bars, class_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.title('Class Distribution: Phishing vs Legitimate', fontsize=14, fontweight='bold')
plt.ylabel('Number of Websites', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: outputs/01_class_distribution.png")

# --- 2c. Correlation Heatmap ---
plt.figure(figsize=(18, 14))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, cmap='coolwarm', center=0,
    annot=False, linewidths=0.5,
    cbar_kws={'shrink': 0.8, 'label': 'Correlation'}
)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: outputs/02_correlation_heatmap.png")

# --- 2d. Top correlated features with target ---
target_corr = corr['Result'].drop('Result').abs().sort_values(ascending=False)
print(f"\n[INFO] Top 10 Features Most Correlated with Phishing:")
for i, (feat, val) in enumerate(target_corr.head(10).items(), 1):
    print(f"   {i:2d}. {feat:<30s} {val:.4f}")

plt.figure(figsize=(10, 6))
top_features = target_corr.head(10)
colors_bar = sns.color_palette('viridis', len(top_features))
bars = plt.barh(
    top_features.index[::-1], top_features.values[::-1],
    color=colors_bar[::-1], edgecolor='black', linewidth=0.5
)
plt.xlabel('Absolute Correlation with Target', fontsize=12)
plt.title('Top 10 Features Most Correlated with Phishing', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/03_top_features_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: outputs/03_top_features_correlation.png")

# --- 2e. Distribution of key features by class ---
key_features = ['sslfinal_state', 'url_of_anchor', 'web_traffic',
                'having_sub_domain', 'url_length', 'having_ip_address']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, feat in zip(axes.flatten(), key_features):
    for cls, color, label in [(-1, '#e74c3c', 'Phishing'), (1, '#2ecc71', 'Legitimate')]:
        subset = df[df['Result'] == cls][feat]
        ax.hist(subset, alpha=0.6, color=color, label=label, bins=5, edgecolor='black')
    ax.set_title(feat, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
plt.suptitle('Feature Distributions: Phishing vs Legitimate', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/04_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: outputs/04_feature_distributions.png")

# =============================================================================
# STEP 3 - DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Data Preprocessing")
print("=" * 60)

# Train/Test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set:     {X_test.shape[0]} samples")

# Feature scaling (needed for Logistic Regression & KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   [OK] StandardScaler applied (for LR & KNN)")

# =============================================================================
# STEP 4 - TRAIN ML MODELS
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Training ML Models")
print("=" * 60)

# Store results
results = {}

# -- Model 1: Logistic Regression ---------------------------------------------
print("\n>> Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
results['Logistic Regression'] = {
    'model': lr,
    'predictions': y_pred_lr,
    'uses_scaler': True
}
print("   [OK] Done!")

# -- Model 2: Decision Tree ---------------------------------------------------
print("\n>> Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, max_depth=10)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
results['Decision Tree'] = {
    'model': dt,
    'predictions': y_pred_dt,
    'uses_scaler': False
}
print("   [OK] Done!")

# -- Model 3: Random Forest ---------------------------------------------------
print("\n>> Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results['Random Forest'] = {
    'model': rf,
    'predictions': y_pred_rf,
    'uses_scaler': False
}
print("   [OK] Done!")

# -- Model 4: K-Nearest Neighbors ---------------------------------------------
print("\n>> Training KNN (k=5)...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
results['KNN'] = {
    'model': knn,
    'predictions': y_pred_knn,
    'uses_scaler': True
}
print("   [OK] Done!")

# =============================================================================
# STEP 5 - EVALUATION & COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Model Evaluation & Comparison")
print("=" * 60)

# Calculate metrics for all models
metrics_data = []
for name, info in results.items():
    y_pred = info['predictions']
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics_data.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df = metrics_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)

# Print comparison table
print("\n" + "-" * 72)
print(f"{'Model':<25s} {'Accuracy':>10s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}")
print("-" * 72)
for _, row in metrics_df.iterrows():
    print(f"{row['Model']:<25s} {row['Accuracy']:>10.4f} {row['Precision']:>10.4f} "
          f"{row['Recall']:>10.4f} {row['F1-Score']:>10.4f}")
print("-" * 72)

best_model_name = metrics_df.iloc[0]['Model']
print(f"\n** Best Model: {best_model_name} (highest F1-Score: {metrics_df.iloc[0]['F1-Score']:.4f})")

# Save metrics to CSV
metrics_df.to_csv('outputs/model_comparison.csv', index=False)
print("   [OK] Saved: outputs/model_comparison.csv")

# --- 5a. Accuracy Comparison Bar Chart ---
plt.figure(figsize=(10, 6))
colors_models = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6']
bars = plt.bar(
    metrics_df['Model'], metrics_df['Accuracy'] * 100,
    color=colors_models[:len(metrics_df)],
    edgecolor='black', linewidth=0.8
)
for bar, acc in zip(bars, metrics_df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{acc*100:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylim(80, 100)
plt.tight_layout()
plt.savefig('outputs/05_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: outputs/05_accuracy_comparison.png")

# --- 5b. All Metrics Grouped Bar Chart ---
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metrics_df))
width = 0.2
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_metrics = ['#3498db', '#e67e22', '#2ecc71', '#e74c3c']

for i, (metric, color) in enumerate(zip(metric_names, colors_metrics)):
    bars = ax.bar(x + i * width, metrics_df[metric] * 100, width,
                  label=metric, color=color, edgecolor='black', linewidth=0.5)

ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Model Performance Comparison - All Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_df['Model'], fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(80, 100)
plt.tight_layout()
plt.savefig('outputs/06_all_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: outputs/06_all_metrics_comparison.png")

# --- 5c. Confusion Matrices ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, info) in zip(axes.flatten(), results.items()):
    cm = confusion_matrix(y_test, info['predictions'])
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Phishing', 'Legitimate'],
        yticklabels=['Phishing', 'Legitimate']
    )
    acc = accuracy_score(y_test, info['predictions'])
    ax.set_title(f'{name}\n(Accuracy: {acc*100:.2f}%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
plt.suptitle('Confusion Matrices - All Models', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/07_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("   [OK] Saved: outputs/07_confusion_matrices.png")

# --- 5d. Detailed Classification Reports ---
print("\n[INFO] Detailed Classification Reports:")
for name, info in results.items():
    print(f"\n{'-'*50}")
    print(f"  {name}")
    print(f"{'-'*50}")
    print(classification_report(y_test, info['predictions'],
                                target_names=['Phishing', 'Legitimate']))

# --- 5e. Feature Importance (from Random Forest) ---
feature_importance = pd.Series(
    rf.feature_importances_, index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 8))
top_n = 15
colors_fi = sns.color_palette('viridis', top_n)
plt.barh(
    feature_importance.head(top_n).index[::-1],
    feature_importance.head(top_n).values[::-1],
    color=colors_fi[::-1], edgecolor='black', linewidth=0.5
)
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 15 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/08_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n   [OK] Saved: outputs/08_feature_importance.png")

# =============================================================================
# STEP 6 - SAVE BEST MODEL
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Saving Models")
print("=" * 60)

# Save all models
joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(dt, 'models/decision_tree.pkl')
joblib.dump(lr, 'models/logistic_regression.pkl')
joblib.dump(knn, 'models/knn.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Save feature names
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

# Save metrics
metrics_df.to_csv('models/metrics.csv', index=False)

print("   [OK] Saved: models/random_forest.pkl")
print("   [OK] Saved: models/decision_tree.pkl")
print("   [OK] Saved: models/logistic_regression.pkl")
print("   [OK] Saved: models/knn.pkl")
print("   [OK] Saved: models/scaler.pkl")
print("   [OK] Saved: models/feature_names.pkl")
print("   [OK] Saved: models/metrics.csv")

# =============================================================================
# DONE!
# =============================================================================
print("\n" + "=" * 60)
print("ALL DONE! Summary:")
print("=" * 60)
print(f"   Dataset: {X.shape[0]} websites x {X.shape[1]} features")
print(f"   Models trained: 4")
print(f"   Best model: {best_model_name}")
print(f"   Charts saved to: outputs/")
print(f"   Models saved to: models/")
print(f"\n   Next step -> Run the Streamlit app:")
print(f"      streamlit run app.py")
print("=" * 60)
