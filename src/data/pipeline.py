import html
import re

import pandas as pd
from sklearn.model_selection import train_test_split

_EARLY_ACCESS = re.compile(r"^\s*early\s+access\s+review\s*", re.IGNORECASE)
_BBCODE = re.compile(r"\[/?[a-zA-Z0-9*=]+\]")
_URL = re.compile(r"https?://\S+|www\.\S+")
_REPEAT = re.compile(r"(.)\1{2,}")
_PUNCT = re.compile(r"([!?.,])")
_WS = re.compile(r"\s+")
_ALNUM = re.compile(r"[A-Za-z0-9]")

def normalize_text(text: str) -> str:
    """Steam 리뷰 노이즈 제거: HTML 엔티티 복원, Early Access 보일러플레이트·BBCode·URL 제거,
    3+ 반복문자 축약, 구두점 분리."""
    t = html.unescape(text)
    t = _EARLY_ACCESS.sub(" ", t)
    t = _BBCODE.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _REPEAT.sub(r"\1\1", t)
    t = _PUNCT.sub(r" \1 ", t)
    return _WS.sub(" ", t).strip()

def clean_reviews(df: pd.DataFrame, text_col: str, label_col: str,
                  normalize: bool = False, min_words: int = 0) -> pd.DataFrame:
    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    out = out.dropna(subset=["text", "label"])
    out["text"] = out["text"].astype(str).str.strip()
    out = out[out["text"].str.len() > 0]
    # Steam review_score: 1=추천, -1=비추천 → 1/0
    out["label"] = (out["label"].astype(int) > 0).astype(int)
    if normalize:
        out["text"] = out["text"].map(normalize_text)
        # 영숫자가 하나도 없는 리뷰(이모지·기호만)는 감성 신호가 없어 제거
        out = out[out["text"].str.contains(_ALNUM, regex=True)]
    if min_words:
        out = out[out["text"].str.split().str.len() >= min_words]
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
