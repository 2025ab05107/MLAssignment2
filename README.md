# MLAssignment2

## Problem Statement
Build multiple ML classifiers and deploy using Streamlit.

## Dataset Description
Dataset source: https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
Instances: 13339 (Originally 30000)*  
Features: 24  
Target: default.payment.next.month
* Few records removed due to the large size and it was not handling the "Random Tree" in dev env.  

## Models Used

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.7854 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 |
| Decision Tree | 0.2146 | 0.5 | 0.2146 | 1.0 | 0.3533 | 0.0 |
| KNN | 0.2239 | 0.4987 | 0.2153 | 0.9899 | 0.3537 | 0.0161 |
| Naive Bayes | 0.2146 | 0.5 | 0.2146 | 1.0| 0.3533 | 0.0 |
| Random Forest | 0.7434 | 0.6583 | 0.4104 | 0.4486 | 0.4286 | 0.264 |
| XGBoost | 0.6795 | 0.6635 | 0.3466 | 0.5577 | 0.4275 | 0.2335 |

## Observations

| Model | Observation | Remark |
|------|-------------|
| Logistic Regression | Logistic Regression failed to detect default cases and is unsuitable despite high accuracy due to severe class imbalance. | Achieved the highest accuracy (78.54%), but produced zero precision, recall, F1, and MCC, indicating it predicted only the majority class. AUC of 0.5 confirms no discriminative power. |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Random Forest | |
| XGBoost | |
