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
|------|-------------|---------|
| Logistic Regression | Logistic Regression failed to detect default cases and is unsuitable despite high accuracy due to severe class imbalance. | * Achieved the highest accuracy (78.54%), but produced zero precision, recall, F1, and MCC, indicating it predicted only the majority class. * AUC of 0.5 confirms no discriminative power. |
| Decision Tree | Decision Tree overfits and classifies most samples as defaulters, leading to poor generalization. | *Low accuracy (21.46%) but recall of 1.0, meaning it detected all defaulters. * Very low precision and MCC indicate large number of false positives. * AUC of 0.5 suggests random-like performance. |
| KNN | KNN captures minority class but lacks stability and overall predictive strength. | * Similar behavior to Decision Tree with very high recall (0.9899) but poor accuracy. * Slight improvement in MCC (0.0161), still very weak. |
| Naive Bayes | Naive Bayes detects defaults but produces excessive false alarms. | * Same pattern as Decision Tree: perfect recall but low accuracy and zero MCC. * Strong assumption of feature independence limits its performance. |
| Random Forest | Random Forest provides the best trade-off between precision and recall and is the most reliable model in this study. | * Strong balanced performance with Accuracy 74.34%, F1 0.4286, MCC 0.264 (highest). * AUC of 0.6583 shows good class separation. |
| XGBoost | XGBoost shows strong learning capability but underperforms Random Forest in overall balance. | Good recall (0.5577) and AUC (0.6635 – highest), but slightly lower F1 and MCC than Random Forest. |
