# Heart Disease Prediction — CodeAlpha ML Internship (Task 4)

## Objective
Predict the presence of heart disease in patients using classification algorithms on structured medical data.

## Dataset
[Heart Disease UCI (Cleveland)](https://archive.ics.uci.edu/dataset/45/heart+disease) — 297 patients, 13 clinical features (age, chest pain type, cholesterol, max heart rate, etc.), fetched via the `ucimlrepo` package.

## Approach
- Preprocessed and cleaned the dataset (removed rows with missing values)
- Binarized the target (0 = no disease, 1 = disease present)
- Trained and compared three classifiers: Logistic Regression, Random Forest, XGBoost
- Selected **Random Forest** as the best model based on accuracy and ROC-AUC

## Results
| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 83% | 0.950 |
| **Random Forest** | **87%** | **0.946** |
| XGBoost | 87% | 0.892 |

## Key Insight
[Fill in top 3-4 features from your feature importance chart, e.g. thal, cp, ca, oldpeak were the strongest predictors]

## Files
- `train.py` — data loading, preprocessing, training script
- `app.py` — Streamlit demo application (patient data input form)
- `heart_disease_model.pkl` — trained Random Forest model
- `scaler.pkl` — fitted StandardScaler for input preprocessing

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Demo
[Add screenshot of the app here]

## Disclaimer
This is a machine learning demo built for educational purposes as part of the CodeAlpha internship. It is not a substitute for professional medical diagnosis.

## Author
Fahad — CodeAlpha ML Internship
