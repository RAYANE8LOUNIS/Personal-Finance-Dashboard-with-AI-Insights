# src/ai_models.py
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def train_kmeans(df, n_clusters=5, random_state=42):
    """
    Clusters transactions to find common groups (e.g., recurring small food purchases,
    large shopping, bills).
    """
    # We'll use amount_log + category (one-hot)
    features = df[['amount_log', 'category']].copy()
    ct = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['amount_log']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['category'])
    ])
    pipe = Pipeline([
        ('pre', ct),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=random_state))
    ])
    pipe.fit(features)
    labels = pipe.named_steps['kmeans'].labels_
    df = df.copy()
    df['cluster'] = labels
    return pipe, df

def detect_anomalies(df, contamination=0.02, random_state=42):
    """
    Use IsolationForest on numerical features to detect unusual transactions.
    """
    # numeric features for anomaly detection:
    features = df[['amount_log']].values
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X)
    preds = iso.predict(X)  # -1 anomaly, 1 normal
    df = df.copy()
    df['is_anomaly'] = preds == -1
    # anomaly score (the smaller, the more anomalous)
    df['anomaly_score'] = iso.decision_function(X)
    return iso, scaler, df
