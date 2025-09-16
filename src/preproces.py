# src/preprocess.py
import pandas as pd
import numpy as np

def load_transactions(path, date_col='date'):
    df = pd.read_csv(path)
    # parse dates
    df[date_col] = pd.to_datetime(df[date_col])
    return df

def clean_categories(df, cat_col='category'):
    # Lowercase & strip; you can expand mapping rules here
    df[cat_col] = df[cat_col].astype(str).str.strip().str.title()
    # Example normalization (expand as needed)
    mapping = {
        'Food & Drinks': 'Food',
        'Restaurants': 'Food',
        'Transport': 'Transport',
    }
    df[cat_col] = df[cat_col].replace(mapping)
    return df

def feature_engineer(df, date_col='date', amount_col='amount'):
    df = df.copy()
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.to_period('M').astype(str)
    df['day'] = df[date_col].dt.day
    df['weekday'] = df[date_col].dt.day_name()
    # keep amount positive
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0.0).abs()
    # log-transform for skewed distribution; add small epsilon
    df['amount_log'] = np.log1p(df[amount_col])
    # optional: bucket amounts
    bins = [0, 5, 20, 50, 100, 500, 10000]
    labels = ['very_small','small','medium','large','very_large','huge']
    df['amount_bucket'] = pd.cut(df[amount_col], bins=bins, labels=labels, include_lowest=True)
    return df

def preprocess_pipeline(path):
    df = load_transactions(path)
    df = clean_categories(df)
    df = feature_engineer(df)
    return df
  
  