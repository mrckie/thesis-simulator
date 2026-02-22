import pandas as pd

def load_summary():
    return pd.read_csv("data/results_summary.csv")

def load_curves():
    return pd.read_csv("data/training_curves.csv")

def load_confusion():
    return pd.read_csv("data/confusion_matrix.csv")