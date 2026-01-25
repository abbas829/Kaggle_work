import pandas as pd

try:
    url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
    df = pd.read_csv(url)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    
    cat_cols = df.select_dtypes('object').columns.tolist()
    num_cols = [col for col in df.columns if col not in cat_cols and col != 'Churn']
    
    print("CATEGORICAL_COLUMNS:", cat_cols)
    print("NUMERICAL_COLUMNS:", num_cols)
    
    # Also print unique values for categorical columns to create selectboxes
    print("\nUNIQUE_VALUES:")
    for col in cat_cols:
        print(f"{col}: {df[col].unique().tolist()}")

except Exception as e:
    print(f"Error: {e}")
