# src/analysis.py
import pandas as pd
import numpy as np

def monthly_spending(df):
    # returns Series indexed by period string like '2025-01'
    s = df.groupby('month')['amount'].sum().sort_index()
    return s

def spending_by_category_month(df):
    # pivot table: rows=month, columns=category
    table = df.pivot_table(index='month', columns='category', values='amount', aggfunc='sum', fill_value=0)
    return table

def compare_last_two_months(df):
    # compute percent change per category between last and previous month
    table = spending_by_category_month(df)
    if table.shape[0] < 2:
        return {}
    last = table.iloc[-1]
    prev = table.iloc[-2]
    pct_change = ((last - prev) / prev.replace({0: np.nan})) * 100
    # handle zeros in prev: if prev==0 and last>0 -> define as 100%+ (or mark as new spend)
    result = {}
    for cat in last.index:
        prev_v = prev[cat]
        last_v = last[cat]
        if prev_v == 0:
            if last_v == 0:
                change = 0.0
                note = 'no change'
            else:
                change = np.inf
                note = f'New spending of {last_v:.2f} in {cat} (no previous month spending)'
        else:
            change = ((last_v - prev_v) / prev_v) * 100.0
            note = f'{change:.1f}%'
        result[cat] = {'prev': prev_v, 'last': last_v, 'change_pct': change, 'note': note}
    return result

def generate_insights(df, anomalies_df=None, threshold_pct=10):
    """
    Produce human-readable insights:
      - Large changes in categories
      - Top spending categories
      - Flag anomalies
    """
    insights = []
    # monthly totals
    monthly = monthly_spending(df)
    if len(monthly) >= 1:
        insights.append(f"Total spending in last month ({monthly.index[-1]}): {monthly.iloc[-1]:.2f}")
    if len(monthly) >= 2:
        pct = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2]) * 100
        insights.append(f"Change vs previous month: {pct:.1f}%")
    # category changes
    changes = compare_last_two_months(df)
    for cat, info in changes.items():
        ch = info['change_pct']
        if ch == np.inf:
            insights.append(f"New spending in {cat}: {info['last']:.2f} (no spending last month).")
        elif abs(ch) >= threshold_pct:
            updown = 'increased' if ch > 0 else 'decreased'
            insights.append(f"{cat} spending {updown} by {ch:.1f}% (from {info['prev']:.2f} to {info['last']:.2f}).")
    # top categories last month
    bycat = df.groupby('category')['amount'].sum().sort_values(ascending=False)
    top = bycat.head(3)
    insights.append("Top categories overall: " + ", ".join([f"{i} ({top[i]:.0f})" if isinstance(i, str) else str(i) for i in top.index]))
    # anomalies
    if anomalies_df is not None and not anomalies_df.empty:
        insights.append(f"Detected {len(anomalies_df)} unusual transactions (possible anomalies).")
    return insights
