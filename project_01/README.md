# ChurnGuard: Customer Churn Prediction

A machine learning application that predicts customer churn for telco companies using ensemble models.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (First Time Only)
Run this command once to train and save the prediction models:
```bash
python setup_models.py
```

This will:
- Download the Telco Customer Churn dataset
- Preprocess the data
- Train multiple models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
- Create an ensemble (Stacking) model
- Train a revenue loss predictor
- Train an ARIMA forecast model
- Save all models to `churn_model.pkl`

### 3. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

- **Single Customer Prediction**: Enter customer details to get churn probability
- **Batch Prediction**: Upload CSV file for batch predictions
- **Revenue Impact**: Estimate potential revenue loss if customer churns
- **Churn Forecast**: Visualize churn trends over time

## File Structure

- `app.py` - Streamlit web application
- `setup_models.py` - Model training script
- `train_models.py` - Alternative training script
- `data_prep_and_eda.ipynb` - EDA and data exploration notebook
- `requirements.txt` - Python dependencies

## Troubleshooting

**FileNotFoundError: churn_model.pkl not found**
- Run `python setup_models.py` to train and save the models

**ModuleNotFoundError**
- Make sure dependencies are installed: `pip install -r requirements.txt`
