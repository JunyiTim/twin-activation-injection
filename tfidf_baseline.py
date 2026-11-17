"""Bag-of-words (TF-IDF) baseline: can the participants' writing alone separate prime from baseline?
Uses the public Mega Study raw responses (.dat/affective_priming/raw_data/response.csv).
20 random stratified 80/20 splits; logistic regression, C=100.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("response.csv", low_memory=False, skiprows=[1, 2])
for domain, col in (("gratitude", "g_writing"), ("lack of control", "c_writing")):
    txt = df[col].astype(str).str.strip()
    d = df[df[col].notna() & (txt != "") & (txt != "nan") & df.cond.isin(["prime", "control"])]
    X, y = d[col].astype(str).str.strip().values, (d.cond == "prime").astype(int).values
    accs = []
    for s in range(20):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=s, stratify=y)
        v = TfidfVectorizer()
        m = LogisticRegression(C=100, max_iter=2000).fit(v.fit_transform(Xtr), ytr)
        accs.append(m.score(v.transform(Xte), yte))
    print(f"{domain}: n+={y.sum()} n-={(1 - y).sum()} acc mean={np.mean(accs):.3f} range=[{min(accs):.3f},{max(accs):.3f}]")
