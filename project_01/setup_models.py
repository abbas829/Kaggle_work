#!/usr/bin/env python3
"""
Setup script to train and save churn prediction models.
Run this once before using app.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from statsmodels.tsa.arima.model import ARIMA

print("Loading data...")
url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
df = pd.read_csv(url)

print("Preprocessing data...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

cat_cols = df.select_dtypes('object').columns.tolist()
num_cols = [col for col in df.columns if col not in cat_cols and col != 'Churn']

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

print("Transforming data...")
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print("Training models...")
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train_prep, y_train)

tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train_prep, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_prep, y_train)

gbm_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gbm_model.fit(X_train_prep, y_train)

estimators = [('rf', rf_model), ('gbm', gbm_model), ('tree', tree_model)]
stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stack_model.fit(X_train_prep, y_train)

print("Training revenue loss predictor...")
df['Predicted_Prob'] = stack_model.predict_proba(preprocessor.transform(X))[:, 1]
rev_X = df[['MonthlyCharges', 'tenure']]
rev_y = df['MonthlyCharges'] * df['Predicted_Prob']
rev_model = LinearRegression()
rev_model.fit(rev_X, rev_y)

print("Training ARIMA model...")
churn_ts = df.groupby('tenure')['Churn'].mean()
arima_model = ARIMA(churn_ts, order=(1,1,1))
arima_fit = arima_model.fit()

print("Saving models...")
with open('churn_model.pkl', 'wb') as f:
    pickle.dump({
        'preprocessor': preprocessor,
        'stack_model': stack_model,
        'rev_model': rev_model,
        'arima_fit': arima_fit
    }, f)

print("✅ Models trained and saved as churn_model.pkl")
print(f"Training accuracy: {stack_model.score(X_train_prep, y_train):.3f}")
print(f"Testing accuracy: {stack_model.score(X_test_prep, y_test):.3f}")
