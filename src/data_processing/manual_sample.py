"""
manual_sample.py

Purpose


"""

import pandas as pd

df = pd.read_csv("data/processed/icu_corpus.csv")

sample_30 = df.sample(n=30, random_state=42)

sample_30.to_csv("data/sample/manual_sample_30.csv", index=False)