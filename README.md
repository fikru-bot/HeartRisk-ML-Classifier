# 🫀 HeartRisk ML Classifier

A machine learning-based classification project to predict the likelihood of heart disease in patients using medical diagnostic attributes.

## 📌 Project Overview

This project applies classification algorithms to medical data to identify the risk of heart disease. It leverages scikit-learn for training and evaluating multiple models and includes comprehensive exploratory data analysis (EDA), feature importance visualization, and model performance comparison.

## 🔍 Objectives

- Perform exploratory data analysis (EDA) to understand trends and patterns.
- Train and evaluate multiple machine learning models (KNN, Logistic Regression, Random Forest).
- Tune model hyperparameters to improve performance.
- Visualize model performance using metrics such as ROC curve, confusion matrix, and classification report.
- Identify the most influential features in predicting heart disease.

## 🧰 Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn (for modeling and evaluation)

## 📂 Dataset

- Source: [Kaggle – Heart Disease Classification Dataset](https://www.kaggle.com/datasets/sumaiyatasmeem/heart-disease-classification-dataset)
- Features include: `age`, `sex`, `cp`, `chol`, `thalach`, `oldpeak`, etc.
- Target variable: `target` (1 = heart disease, 0 = no heart disease)

## ⚙️ Model Development Workflow

1. **EDA & Preprocessing**
   - Data inspection, type conversion, and missing value handling.
   - Categorical encoding and scaling as needed.

2. **Model Training**
   - Models trained: `KNN`, `Logistic Regression`, and `Random Forest`.

3. **Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrix and ROC curve
   - Cross-validation metrics for robustness

4. **Feature Importance**
   - Visualized to understand feature contribution to predictions.

## 🧪 Results

- Logistic Regression and Random Forest both achieved high performance.
- ROC AUC score of ~0.92 using the best-tuned Logistic Regression model.
- Features like chest pain type (`cp`), slope, and restecg were most influential.

## 📊 Visualizations

- Correlation heatmap
- Bar charts for model comparison
- ROC curve and confusion matrix
- Feature importance ranking

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/fikru-bot/HeartRisk-ML-Classifier.git
   cd HeartRisk-ML-Classifier
