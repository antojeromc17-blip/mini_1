# 🛡️ Phishing Website Detection Using Machine Learning

> **KTU S3 CSE AI & ML Mini Project**  
> **Course:** CS201 / CST201 (AI & Machine Learning Mini Project)  
> **Topic:** Binary Classification for Cybersecurity  

---

## 📋 Problem Statement

Phishing websites are fraudulent web pages designed by cyber attackers to deceive victims into submitting sensitive information (such as credentials, credit card details, or personal data) by impersonating legitimate institutions.  

This project develops and compares supervised machine learning models to classify web pages as **Legitimate (1)** or **Phishing (-1)** based on 30 distinct URL and page characteristics, achieving high detection accuracy and low false-negative rates.

---

## 📊 Dataset Overview

- **Source:** [UCI Machine Learning Repository — Phishing Websites Dataset (ID: 327)](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- **Total Samples:** 11,055 websites
- **Target Distribution:**
  - Legitimate (1): 6,157 websites (55.7%)
  - Phishing (-1): 4,898 websites (44.3%)
- **Total Features:** 30 categorical & numeric indicators (encoded as `-1`, `0`, `1`)
- **Missing Values:** None (0 missing records)

### Key Features Identified by Model (Top 5):
1. **`sslfinal_state`**: Validity, authority, and age of SSL/TLS certificate.
2. **`url_of_anchor`**: Percentage of anchor (`<a>`) tags pointing to abnormal/external domains.
3. **`prefix_suffix`**: Presence of hyphens (`-`) in domain name.
4. **`web_traffic`**: Website popularity rank (Alexa/traffic records).
5. **`having_sub_domain`**: Number of sub-domain dots in the URL structure.

---

## 🤖 Machine Learning Algorithms & Results

All models were evaluated on an 80/20 stratified train-test split (8,844 training samples, 2,211 test samples).

| Model | Accuracy | Precision | Recall | F1-Score | Status |
|---|---|---|---|---|---|
| **Random Forest** | **97.20%** | **96.28%** | **98.78%** | **0.9751** | ⭐ **Best Model** |
| **K-Nearest Neighbors (KNN)** | 94.89% | 95.01% | 95.86% | 0.9543 | High Performance |
| **Decision Tree** | 94.80% | 95.74% | 94.88% | 0.9531 | Interpretable |
| **Logistic Regression** | 92.85% | 92.34% | 95.05% | 0.9367 | Baseline Linear |

> **Key Finding:** Random Forest achieved the highest F1-Score (**97.51%**) and highest Recall (**98.78%**), ensuring minimal phishing sites escape detection (low False Negatives).

---

## 📁 Project Structure

```
ML_MINI_PROJECT/
├── data/
│   └── phishing_dataset.csv         ← Full dataset (11,055 rows)
├── notebooks/
│   └── phishing_analysis.ipynb      ← Complete Jupyter Notebook for submission
├── models/
│   ├── random_forest.pkl            ← Best trained ensemble model
│   ├── decision_tree.pkl            ← Trained decision tree model
│   ├── logistic_regression.pkl      ← Trained logistic regression
│   ├── knn.pkl                      ← Trained KNN model
│   ├── scaler.pkl                   ← StandardScaler for feature normalization
│   ├── feature_names.pkl            ← Feature column names list
│   └── metrics.csv                  ← Evaluation metrics summary
├── outputs/
│   ├── 01_class_distribution.png     ← Class balance visualization
│   ├── 02_correlation_heatmap.png    ← Feature correlation matrix
│   ├── 03_top_features_correlation.png← Top features correlation bar chart
│   ├── 04_feature_distributions.png  ← Phishing vs Legit feature histograms
│   ├── 05_accuracy_comparison.png    ← Model accuracy comparison chart
│   ├── 06_all_metrics_comparison.png ← Multi-metric comparison chart
│   ├── 07_confusion_matrices.png     ← 2x2 confusion matrices for all models
│   ├── 08_feature_importance.png     ← Gini feature importance bar chart
│   └── model_comparison.csv          ← Raw evaluation metrics
├── train_model.py                   ← Model training & evaluation script
├── app.py                           ← Interactive Streamlit web demo UI
├── requirements.txt                 ← Python library dependencies
└── README.md                        ← Project documentation
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Training Pipeline
```bash
python train_model.py
```
*Generates all models in `models/` and charts in `outputs/`.*

### 3. Launch the Interactive Web App
```bash
streamlit run app.py
```
*Opens in browser at `http://localhost:8501`.*

---

## 🖥️ Web App Features
- **Intuitive Inputs:** Grouped by URL structure, security certificates, and webpage content.
- **Real-Time Classification:** Instant prediction badge with confidence percentage.
- **Model Selector:** Switch between Random Forest, Decision Tree, Logistic Regression, and KNN.
- **Interactive Performance Dashboard:** Live accuracy comparison, metrics table, and feature importance.

---

## 🎓 Viva Questions & Answers

1. **Why is Recall more important than Precision in phishing detection?**  
   *A False Negative means a phishing website slipped through, putting users at risk of credentials theft. A high Recall (98.78%) ensures almost all phishing websites are captured.*

2. **Why does Random Forest outperform a single Decision Tree?**  
   *Random Forest is an ensemble of 100 decision trees trained on bootstrap samples with random feature subsets. It averages predictions, reducing variance and mitigating overfitting.*

3. **Why do we need feature scaling for KNN and Logistic Regression, but not Random Forest?**  
   *KNN relies on Euclidean distance and Logistic Regression uses gradient descent; large value ranges distort distances and gradient steps. Tree algorithms make orthogonal threshold splits, which are scale-invariant.*
