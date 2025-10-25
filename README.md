# 💳 Credit Card Fraud Detection

## 📘 Overview
This project aims to detect fraudulent credit card transactions using **Machine Learning** techniques.  
Due to the high imbalance between genuine and fraudulent transactions, identifying fraud accurately is challenging — this notebook demonstrates data analysis, preprocessing, model training, and evaluation to tackle that issue effectively.  

We apply a range of models including **Logistic Regression**, **Random Forest**, and **XGBoost**, combined with **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset.  
Visualizations are created using **Seaborn** and **Matplotlib** to provide insights into data distribution and model performance.

---

## 📊 Dataset
The dataset used is the **Credit Card Fraud Detection Dataset** from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
It contains transactions made by European credit cardholders in September 2013.

**Key Facts:**
- 🧾 **Total Transactions:** 284,807  
- ⚠️ **Fraudulent Transactions:** 492 (~0.17%)  
- 🧮 **Features:** 30 total (28 anonymized using PCA + Time + Amount + Class)  
- 🎯 **Target Variable:** `Class` — 0 for non-fraud, 1 for fraud  

---

## ⚙️ Installation & Setup

Clone the repository and install dependencies:


```bash
git clone https://github.com/yourusername/CCFraudDetection.git
cd CCFraudDetection
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
```
---
## 📈 Results Summary
| Model               | Precision | Recall   | F1-Score | ROC-AUC  |
| :------------------ | :-------- | :------- | :------- | :------- |
| Logistic Regression | 0.93      | 0.80     | 0.86     | 0.95     |
| Random Forest       | 0.97      | 0.89     | 0.93     | 0.99     |
| XGBoost             | **0.98**  | **0.91** | **0.94** | **0.99** |

## 📊 Insights

The dataset is highly imbalanced, making recall (fraud detection rate) crucial.

SMOTE helps the model generalize better on minority classes.

XGBoost consistently outperforms other models in accuracy and recall.

Ensemble methods handle complex relationships better than linear models.
