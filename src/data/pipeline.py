import pandas as pd
from sklearn.model_selection import train_test_split

def clean_reviews(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    out = out.dropna(subset=["text", "label"])
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"].str.len() > 0]
    # Steam review_score: 1=추천, -1=비추천 → 1/0
    out["label"] = (out["label"].astype(int) > 0).astype(int)
    out = out.drop_duplicates().reset_index(drop=True)
    return out

def split_data(df, seed=42):
    train, temp = train_test_split(
        df, test_size=0.3, random_state=seed, stratify=df["label"])
    val, test = train_test_split(
        temp, test_size=0.5, random_state=seed, stratify=temp["label"])
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))
