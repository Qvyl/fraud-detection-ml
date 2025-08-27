# Credit Card Fraud Detection

## 📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using machine learning. Since fraud detection is a **highly imbalanced classification problem**, the goal is to build and evaluate models that can effectively identify rare fraudulent cases without being overwhelmed by the majority of non-fraudulent ones.

The dataset used comes from Kaggle’s [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains anonymized features derived from PCA, along with `Amount` and `Class` labels, where:
- `Class = 0` → Legitimate transaction  
- `Class = 1` → Fraudulent transaction  

---

## 📂 Workflow

### 1. Data Loading & Exploration
- Loaded the dataset (`creditcard.csv`)  
- Checked dataset shape, missing values, and descriptive statistics  
- Dropped irrelevant column `Time`  

### 2. Data Preprocessing
- Standardized the `Amount` column using `StandardScaler`  
- Split data into **train/test sets** (80/20) with `stratify` to handle imbalance  

### 3. Modeling with Hyperparameter Tuning
Implemented multiple machine learning models with **`class_weight="balanced"`** to address imbalance:

- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Random Forest Classifier**  
- **Support Vector Classifier (SVC)**  

Each model was optimized using **GridSearchCV** with 5-fold cross-validation:

- Logistic Regression → tuned `C`, `solver`  
- Decision Tree → tuned `criterion`, `max_depth`  
- Random Forest → tuned `n_estimators`, `max_depth`  
- SVC → tuned `C`, `kernel`, `gamma`  

### 4. Evaluation Metrics
- Confusion Matrix (visualized with heatmap)  
- Classification Report (Precision, Recall, F1-score)  
- Accuracy  

⚠️ **Key Focus:** Maximizing **Recall on Class 1 (fraud)**, since catching fraudulent cases is more important than overall accuracy.

---

## 📊 Results
- Logistic Regression with tuned hyperparameters achieved strong recall on fraudulent cases with relatively few false negatives.  
- Decision Tree and Random Forest provided flexibility but risked overfitting.  
- SVC required scaling but performed competitively when optimized.  

Confusion matrices for all tuned models showed the trade-offs between false positives and false negatives.

---
