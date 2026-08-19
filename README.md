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
<img width="591" height="470" alt="image" src="https://github.com/user-attachments/assets/46afb28d-08e0-4324-b86a-772cd26199e0" />


## Key Insight
Thalassemia (thal), chest pain type (cp), maximum heart rate achieved (thalach), and ST depression (oldpeak) were the four strongest predictors, together accounting for nearly half the model's decision weight. This aligns with established clinical risk factors for cardiac disease. On the held-out test set, the model correctly identified 52 of 60 patients (86.7% accuracy), with a slightly higher rate of false negatives (5) than false positives (3) — worth noting since missed disease cases carry higher risk than false alarms in a real diagnostic context. 
<img width="788" height="590" alt="image" src="https://github.com/user-attachments/assets/14b2f323-e577-4c9f-b60f-04ebc271d3ce" />


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
<img width="1919" height="908" alt="image" src="https://github.com/user-attachments/assets/5a4a031d-1c72-427e-befb-deffe73c8274" />


## Disclaimer
This is a machine learning demo built for educational purposes as part of the CodeAlpha internship. It is not a substitute for professional medical diagnosis.

## Author
Fahad — CodeAlpha ML Internship
