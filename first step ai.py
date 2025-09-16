# src/generate_mock.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'transactions.csv')

CATEGORIES = {
    'Food': ['McDonalds', 'Starucks', 'Subway', 'Local Diner'],
    'Transport': ['Uber', 'Bus', 'Train', 'Gas Station'],
    'Shopping': ['Zara', 'H&M', 'Amazon', 'Mall Store'],
    'Bills': ['Electric Co', 'Water Co', 'Internet'],
    'Entertainment': ['Netflix', 'Cinema', 'Spotify'],
    'Health': ['Pharmacy', 'Clinic', 'Dentist'],
    'Other': ['Misc', 'Gift', 'Donation']
}

def sample_amount(cat):
    # reasoned ranges per category (mean, sigma for lognormal)
    means = {
        'Food': 20, 'Transport': 7, 'Shopping': 60, 'Bills': 120,
        'Entertainment': 15, 'Health': 50, 'Other': 25
    }
    mean = means.get(cat, 30)
    # lognormal sampling to simulate skew (more small purchases, fewer large)
    return round(np.random.lognormal(np.log(mean), 0.8), 2)

def generate(start_date='2025-01-01', months=6, seed=42, out=OUT):
    np.random.seed(seed)
    random.seed(seed)
    start = datetime.fromisoformat(start_date)
    rows = []
    days = months * 30
    for d in range(days):
        date = start + timedelta(days=d)
        # variable number of transactions per day
        n = np.random.choice([0,1,2,3], p=[0.4, 0.35, 0.2, 0.05])
        for _ in range(n):
            category = random.choices(list(CATEGORIES.keys()), weights=[0.25,0.15,0.15,0.1,0.1,0.1,0.15])[0]
            merchant = random.choice(CATEGORIES[category])
            amount = sample_amount(category)
            # inject rare anomalies with small probability
            if random.random() < 0.01:
                amount *= random.choice([5,10])  # a big unusual purchase
            desc = f"{merchant} {category}"
            rows.append({'date': date.date().isoformat(), 'category': category, 'amount': amount, 'description': desc})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} transactions -> {out}")

if __name__ == '__main__':
    generate()
