# 🩺 Kidney Stone Prediction using Classification Algorithms

A machine learning project focussed on predicting the presence (or risk) of kidney stones using classification models and clinical / demographic features.

---

## 📌 Project Overview

The project builds a full pipeline: collecting patient data (age, gender, hydration level, diet, clinical test results, etc.), performing exploratory analysis, engineering features, training classification models (e.g., logistic regression, random forest, XGBoost), evaluating performance, and deriving insights for medical decision-making. The goal is to build a reliable system that helps identify individuals at high risk for kidney stones, enabling preventive care.

---

## 🧰 Tech Stack

* **Language:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
* **Environment:** Jupyter Notebook or Google Colab

---

## 🔄 Workflow Summary

### 1. Data Collection

Dataset includes features such as: patient age, gender, previous stone history, fluid intake, urinary-calcium/oxalate levels, BMI, diet type, and a target variable indicating presence or diagnosis of kidney stones.

### 2. Exploratory Data Analysis (EDA)

* Visualisation of stone vs non-stone groups by age, gender, fluid intake, test results
* Correlation matrix between features and target
* Histogram of continuous features; bar plots for categorical features
* Checking for class imbalance (often fewer stone cases than normal)

### 3. Feature Engineering

* Encoding of categorical variables (e.g., gender, diet type) via label or one-hot encoding
* Creation of derived features: e.g., calcium/oxalate ratio, fluid_intake_per_kg, years_since_last_stone
* Scaling of numeric features (StandardScaler or MinMaxScaler) as required
* Split data into training and test sets ensuring stratification if class is imbalanced

### 4. Modeling

* Trained classification algorithms such as:

  * Logistic Regression (baseline)
  * Random Forest Classifier (strong performance)
  * XGBoost/LightGBM for boosted performance
* Hyperparam-tuning via GridSearchCV/RandomizedSearchCV for parameters like max_depth, n_estimators, learning_rate

### 5. Evaluation

* Metrics used: Accuracy, Precision, Recall, F1-Score, ROC-AUC
* Examination of confusion matrix: false negatives (undiagnosed stones) have critical clinical implications
* The best model identifies high-risk patients with high recall while maintaining acceptable precision

### 6. Insights & Application

* Key risk-factors identified: e.g., low fluid intake, high urinary oxalate, previous stone history, diet high in oxalate
* Suggested interventions: increased hydration, dietary adjustments, follow-up screening for high-risk patients
* Model can support healthcare providers in targeting preventative measures and monitoring

---

## 📁 Project Structure

```
Kidney-Stone-Prediction/
│── data/
│   ├── raw/
│   └── processed/
│── notebooks/
│   └── kidney_stone_prediction.ipynb
│── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── model.py
│   └── evaluate.py
│── README.md
│── requirements.txt
```

---

## 📈 Key Findings

* Patients with **history of stones + low hydration** had significantly higher risk predicted by the model
* Feature engineering (especially dietary and urine-test derived ratios) improved discrimination compared to only demographic features
* Tree-based models (Random Forest / Boosting) outperformed logistic regression due to ability to capture non-linear interactions
* Balanced model performance required careful handling of class imbalance (e.g., via class weighting or oversampling)

---

## 🚀 Future Improvements

* Incorporate **time-series data** (e.g., monthly urine tests, fluid intake logs) for dynamic risk modelling
* Explore **ensemble methods** (stacking or blending) to further boost performance
* Deploy model as a web-app for clinicians: input patient features → risk score + recommendations
* Integrate **explainability tools** (e.g., SHAP) so physicians understand which features drove the risk score
* Validate model on external datasets (different clinics or populations) to ensure generalisation

