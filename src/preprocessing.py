import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import dump

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def split_features_target(df):
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include='number').columns.tolist()
    return X, y, cat_cols, num_cols

def build_preprocessor(cat_cols, num_cols):
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    num_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    return ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

def split_data(X, y, test_size=0.2, val_size=0.25, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_preprocessor(preprocessor, path='models/preprocessor.joblib'):
    dump(preprocessor, path)