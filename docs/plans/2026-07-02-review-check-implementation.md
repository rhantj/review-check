# review-check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Steam 게임 리뷰의 긍/부정을 직접 학습한 DL 모델로 분류하고, Qwen2.5-Instruct로 장단점을 요약하며, RAG로 리뷰 기반 Q&A를 제공하는 Gradio 웹앱을 HF Spaces에 배포한다.

**Architecture:** 순수 함수 중심의 5개 컴포넌트(데이터 파이프라인 / DL 분류 / LLM 요약 / RAG / Gradio 앱)를 독립적으로 구현·테스트한다. DL 모델은 LSTM과 DistilBERT 두 가지를 같은 지표로 비교하고, 성능 좋은 쪽을 HF Model Hub에 올려 앱에서 로드한다. LLM·임베딩 호출부는 한 곳에 추상화해 로컬(Ollama)↔배포(HF Inference API) 전환을 쉽게 한다.

**Tech Stack:** Python 3.10, PyTorch, HuggingFace Transformers/Datasets, scikit-learn, sentence-transformers, chromadb/faiss-cpu, Gradio 6, HF Inference API (Qwen2.5-Instruct), pytest.

## Global Constraints

- Python 3.10.6, 기존 `.venv` 사용 (`/Users/gomuseo/Desktop/Python/.venv`).
- 마감: 2026-07-07(화). Phase 1(탭1)이 발표 최소 완성본, Phase 2(RAG 탭2)는 시간 부족 시 제거 가능.
- 비밀값은 `.env`로만 관리, 절대 커밋 금지 (`HF_TOKEN` 등).
- 원본 데이터(csv/tsv)·모델 가중치(.pt/.bin/.pkl)는 커밋 금지 (`.gitignore` 적용됨).
- LLM 모델: `Qwen/Qwen2.5-7B-Instruct` (배포 직전 HF Inference 가용성 확인, 폴백 `google/gemma-2-9b-it`).
- 임베딩 모델: `sentence-transformers/all-MiniLM-L6-v2`.
- 모든 신규 소스는 `src/` 아래, 테스트는 `tests/` 아래, 실행 스크립트는 `scripts/` 아래.
- 커밋 메시지: `<type>: <desc>` (feat/fix/test/chore/docs).

---

## File Structure

```
review-check/
├── app.py                     # Gradio 2탭 앱 (오케스트레이션)
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py              # 상수: 모델 ID, 경로, top-k 등
│   ├── data/pipeline.py       # 정제 + train/val/test 분할
│   ├── models/dataset.py      # torch Dataset + 어휘/토큰화 (LSTM용)
│   ├── models/lstm.py         # LSTM 분류기 정의
│   ├── models/infer.py        # 통합 추론 인터페이스 (predict_sentiment)
│   ├── llm/client.py          # LLM 호출 추상화 (HF Inference / Ollama)
│   ├── llm/summarize.py       # 프롬프트 구성 + 장단점 요약
│   └── rag/
│       ├── index.py           # 리뷰 임베딩 → 벡터스토어 구축
│       └── qa.py              # 검색 + 근거 기반 답변
├── scripts/
│   ├── download_data.py       # Kaggle/HF에서 데이터 받기
│   ├── train_lstm.py          # LSTM 학습 + 평가지표 저장
│   ├── train_distilbert.py    # DistilBERT 파인튜닝 + 평가지표 저장
│   ├── compare_models.py      # 두 모델 지표 비교표 출력
│   └── build_index.py         # 벡터DB 구축 실행
├── tests/
│   ├── test_pipeline.py
│   ├── test_infer.py
│   ├── test_summarize.py
│   └── test_qa.py
└── docs/
    ├── specs/2026-07-02-steam-review-sentiment-rag-design.md
    └── plans/2026-07-02-review-check-implementation.md
```

---

## Task 1: 프로젝트 셋업 (의존성 + config)

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `src/config.py`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/llm/__init__.py`, `src/rag/__init__.py`

**Interfaces:**
- Produces: `src.config` 모듈 — `RANDOM_SEED: int`, `LLM_MODEL_ID: str`, `LLM_FALLBACK_ID: str`, `EMBED_MODEL_ID: str`, `RAG_TOP_K: int`, `LABELS: dict[int,str]`, `MAX_LEN: int`, `DATA_DIR: Path`, `MODEL_DIR: Path`, `VECTOR_DIR: Path`.

- [ ] **Step 1: requirements.txt 작성**

```
torch>=2.12
transformers>=5.12
datasets>=3.0
accelerate>=1.0
scikit-learn>=1.7
sentence-transformers>=5.6
chromadb>=1.5
faiss-cpu>=1.14
gradio>=6.19
huggingface-hub>=0.27
python-dotenv>=1.2
pandas>=2.2
matplotlib>=3.10
pytest>=8.0
```

- [ ] **Step 2: 누락 패키지 설치**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/pip install datasets accelerate pytest`
Expected: `Successfully installed datasets-... accelerate-... pytest-...`

- [ ] **Step 3: `.env.example` 작성**

```
# Hugging Face 토큰 (Inference API + Model Hub 업로드용)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
# 로컬 LLM 테스트 시(선택)
LLM_BACKEND=hf   # hf | ollama
```

- [ ] **Step 4: `src/config.py` 작성**

```python
from pathlib import Path

RANDOM_SEED = 42
MAX_LEN = 256

LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LLM_FALLBACK_ID = "google/gemma-2-9b-it"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K = 5

LABELS = {0: "부정", 1: "긍정"}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
VECTOR_DIR = ROOT / "chroma_store"
```

- [ ] **Step 5: 빈 `__init__.py` 5개 생성** (위 Files 목록 경로)

- [ ] **Step 6: config import 확인**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -c "from src.config import LLM_MODEL_ID, RAG_TOP_K; print(LLM_MODEL_ID, RAG_TOP_K)"`
Expected: `Qwen/Qwen2.5-7B-Instruct 5`

- [ ] **Step 7: 커밋**

```bash
git add requirements.txt .env.example src/
git commit -m "chore: project setup — deps and config"
```

---

## Task 2: 데이터 파이프라인 (정제 + 분할)

**Files:**
- Create: `src/data/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: 없음 (pandas DataFrame 입력)
- Produces:
  - `clean_reviews(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame` — 결측/중복/빈문자열 제거, 컬럼을 `text`,`label`(int 0/1)로 표준화한 새 DataFrame 반환(원본 불변).
  - `split_data(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]` — train/val/test = 0.7/0.15/0.15 계층분할 반환.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_pipeline.py
import pandas as pd
from src.data.pipeline import clean_reviews, split_data

def test_clean_removes_nulls_dupes_and_standardizes():
    df = pd.DataFrame({
        "review_text": ["good", "good", "bad", "", None],
        "review_score": [1, 1, -1, 1, 1],
    })
    out = clean_reviews(df, "review_text", "review_score")
    assert list(out.columns) == ["text", "label"]
    assert set(out["label"].unique()) <= {0, 1}
    assert (out["text"].str.len() > 0).all()
    assert out.duplicated().sum() == 0
    # 원본 불변
    assert len(df) == 5

def test_split_ratios_and_stratify():
    df = pd.DataFrame({"text": [str(i) for i in range(100)],
                       "label": [0, 1] * 50})
    tr, va, te = split_data(df, seed=42)
    assert len(tr) + len(va) + len(te) == 100
    assert 68 <= len(tr) <= 72
    assert set(tr["label"].unique()) == {0, 1}
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: src.data.pipeline`

- [ ] **Step 3: 구현**

```python
# src/data/pipeline.py
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
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/data/pipeline.py tests/test_pipeline.py
git commit -m "feat: data cleaning and stratified split"
```

---

## Task 3: 데이터 다운로드 스크립트

**Files:**
- Create: `scripts/download_data.py`

**Interfaces:**
- Consumes: `src.data.pipeline.clean_reviews`, `split_data`, `src.config.DATA_DIR`
- Produces: `data/{train,val,test}.csv` (컬럼 `text`,`label`)

- [ ] **Step 1: 스크립트 작성** (HF `datasets`의 공개 steam 리뷰 사용, Kaggle 없이 재현 가능)

```python
# scripts/download_data.py
"""HF datasets에서 Steam 리뷰를 받아 정제·분할해 data/에 저장.
사용: python -m scripts.download_data --limit 20000
"""
import argparse
from datasets import load_dataset
import pandas as pd
from src.config import DATA_DIR, RANDOM_SEED
from src.data.pipeline import clean_reviews, split_data

def main(limit: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HuggingFaceH4/steam-reviews" if False else "amazon_polarity")
    # 1차: steam 전용 데이터셋 시도, 실패 시 아래 폴백 사용
    raise SystemExit("데이터셋 확정 후 Step 2에서 교체")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=20000)
    main(p.parse_args().limit)
```

- [ ] **Step 2: 실제 데이터셋 확정 및 반영**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -c "from datasets import load_dataset; d=load_dataset('Xanthius/steam-reviews-2021', split='train[:5]'); print(d.column_names)"`
- 위 명령이 성공하면 그 `column_names`에 맞춰 `text_col`/`label_col`을 지정.
- 실패하면 순서대로 시도: `laion/steam-reviews`, `mteb/steam_reviews`, 최종 폴백 `amazon_polarity`(`content`/`label`, `label 1=positive`).
- 확정된 데이터셋으로 `main`을 아래 형태로 교체:

```python
def main(limit: int):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("<확정된_id>", split=f"train[:{limit}]")
    df = ds.to_pandas()
    clean = clean_reviews(df, "<text_col>", "<label_col>")
    tr, va, te = split_data(clean, seed=RANDOM_SEED)
    tr.to_csv(DATA_DIR / "train.csv", index=False)
    va.to_csv(DATA_DIR / "val.csv", index=False)
    te.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
```

- [ ] **Step 3: 실행해서 데이터 생성**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m scripts.download_data --limit 20000`
Expected: `train=... val=... test=...` 출력, `data/train.csv` 등 생성

- [ ] **Step 4: 커밋** (데이터 자체는 .gitignore로 제외됨)

```bash
git add scripts/download_data.py
git commit -m "feat: dataset download and prepare script"
```

---

## Task 4: DistilBERT 파인튜닝 스크립트

**Files:**
- Create: `scripts/train_distilbert.py`

**Interfaces:**
- Consumes: `data/{train,val,test}.csv`, `src.config.{MODEL_DIR,MAX_LEN,RANDOM_SEED}`
- Produces: `models/distilbert/` (저장된 모델+토크나이저), `models/distilbert/metrics.json` (`{"accuracy":float,"f1":float}`)

- [ ] **Step 1: 스크립트 작성**

```python
# scripts/train_distilbert.py
"""사용: python -m scripts.train_distilbert --epochs 2"""
import argparse, json
import numpy as np, pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, f1_score
from src.config import MODEL_DIR, MAX_LEN, DATA_DIR

MODEL = "distilbert-base-uncased"

def load(split):
    df = pd.read_csv(DATA_DIR / f"{split}.csv")
    return Dataset.from_pandas(df)

def main(epochs):
    tok = AutoTokenizer.from_pretrained(MODEL)
    def enc(b): return tok(b["text"], truncation=True, max_length=MAX_LEN)
    ds = {s: load(s).map(enc, batched=True) for s in ["train", "val", "test"]}
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds),
                "f1": f1_score(p.label_ids, preds)}

    out = MODEL_DIR / "distilbert"
    args = TrainingArguments(
        output_dir=str(out), num_train_epochs=epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        eval_strategy="epoch", save_strategy="no", report_to=[])
    trainer = Trainer(model=model, args=args,
                      train_dataset=ds["train"], eval_dataset=ds["val"],
                      compute_metrics=metrics)
    trainer.train()
    test_metrics = trainer.evaluate(ds["test"])
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    result = {"accuracy": test_metrics["eval_accuracy"], "f1": test_metrics["eval_f1"]}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print("DistilBERT test:", result)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=2)
    main(p.parse_args().epochs)
```

- [ ] **Step 2: 소량으로 스모크 실행** (먼저 데이터 5천건으로 파이프라인 검증)

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m scripts.train_distilbert --epochs 1`
Expected: 학습 진행바 후 `DistilBERT test: {'accuracy': ..., 'f1': ...}`, `models/distilbert/metrics.json` 생성

- [ ] **Step 3: 커밋**

```bash
git add scripts/train_distilbert.py
git commit -m "feat: DistilBERT fine-tuning script"
```

---

## Task 5: LSTM 학습 스크립트 (비교군)

**Files:**
- Create: `src/models/dataset.py`
- Create: `src/models/lstm.py`
- Create: `scripts/train_lstm.py`

**Interfaces:**
- Consumes: `data/*.csv`, `src.config.{MAX_LEN,MODEL_DIR}`
- Produces:
  - `src.models.dataset.build_vocab(texts: list[str], min_freq: int=2) -> dict[str,int]`
  - `src.models.dataset.encode(text: str, vocab: dict, max_len: int) -> list[int]`
  - `src.models.lstm.LSTMClassifier(vocab_size:int, embed_dim=100, hidden=128)` — `forward(x)->logits[B,2]`
  - `models/lstm/metrics.json` (`accuracy`,`f1`), `models/lstm/vocab.json`, `models/lstm/model.pt`

- [ ] **Step 1: dataset 유틸 작성**

```python
# src/models/dataset.py
from collections import Counter

PAD, UNK = 0, 1

def build_vocab(texts, min_freq=2):
    c = Counter(w for t in texts for w in t.lower().split())
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w, f in c.items():
        if f >= min_freq:
            vocab[w] = len(vocab)
    return vocab

def encode(text, vocab, max_len):
    ids = [vocab.get(w, UNK) for w in text.lower().split()][:max_len]
    ids += [PAD] * (max_len - len(ids))
    return ids
```

- [ ] **Step 2: LSTM 모델 작성**

```python
# src/models/lstm.py
import torch, torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 2)

    def forward(self, x):
        e = self.emb(x)
        _, (h, _) = self.lstm(e)
        return self.fc(h[-1])
```

- [ ] **Step 3: 학습 스크립트 작성**

```python
# scripts/train_lstm.py
"""사용: python -m scripts.train_lstm --epochs 3"""
import argparse, json
import pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from src.config import DATA_DIR, MODEL_DIR, MAX_LEN
from src.models.dataset import build_vocab, encode
from src.models.lstm import LSTMClassifier

def tensors(df, vocab):
    X = torch.tensor([encode(t, vocab, MAX_LEN) for t in df["text"]])
    y = torch.tensor(df["label"].tolist())
    return TensorDataset(X, y)

def main(epochs):
    tr = pd.read_csv(DATA_DIR/"train.csv"); te = pd.read_csv(DATA_DIR/"test.csv")
    vocab = build_vocab(tr["text"].tolist())
    dl = DataLoader(tensors(tr, vocab), batch_size=64, shuffle=True)
    model = LSTMClassifier(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for X, y in dl:
            opt.zero_grad(); loss = loss_fn(model(X), y); loss.backward(); opt.step()
        print(f"epoch {ep+1} done")
    model.eval()
    Xte = torch.tensor([encode(t, vocab, MAX_LEN) for t in te["text"]])
    with torch.no_grad():
        preds = model(Xte).argmax(1).tolist()
    result = {"accuracy": accuracy_score(te["label"], preds),
              "f1": f1_score(te["label"], preds)}
    out = MODEL_DIR/"lstm"; out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out/"model.pt")
    (out/"vocab.json").write_text(json.dumps(vocab))
    (out/"metrics.json").write_text(json.dumps(result, indent=2))
    print("LSTM test:", result)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--epochs", type=int, default=3)
    main(p.parse_args().epochs)
```

- [ ] **Step 4: 실행**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m scripts.train_lstm --epochs 3`
Expected: `LSTM test: {'accuracy': ..., 'f1': ...}`, `models/lstm/metrics.json` 생성

- [ ] **Step 5: 커밋**

```bash
git add src/models/dataset.py src/models/lstm.py scripts/train_lstm.py
git commit -m "feat: LSTM baseline classifier and training"
```

---

## Task 6: 모델 비교표

**Files:**
- Create: `scripts/compare_models.py`

**Interfaces:**
- Consumes: `models/lstm/metrics.json`, `models/distilbert/metrics.json`
- Produces: 콘솔 비교표 + `models/comparison.md`

- [ ] **Step 1: 스크립트 작성**

```python
# scripts/compare_models.py
import json
from src.config import MODEL_DIR

def load(name):
    return json.loads((MODEL_DIR/name/"metrics.json").read_text())

def main():
    rows = {"LSTM": load("lstm"), "DistilBERT": load("distilbert")}
    lines = ["| 모델 | Accuracy | F1 |", "|---|---|---|"]
    for name, m in rows.items():
        lines.append(f"| {name} | {m['accuracy']:.4f} | {m['f1']:.4f} |")
    table = "\n".join(lines)
    print(table)
    (MODEL_DIR/"comparison.md").write_text(table + "\n")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m scripts.compare_models`
Expected: 마크다운 비교표 출력, `models/comparison.md` 생성

- [ ] **Step 3: 커밋**

```bash
git add scripts/compare_models.py
git commit -m "feat: model comparison table"
```

---

## Task 7: 통합 추론 인터페이스

**Files:**
- Create: `src/models/infer.py`
- Test: `tests/test_infer.py`

**Interfaces:**
- Consumes: `models/distilbert/`
- Produces:
  - `SentimentClassifier(model_dir: str | Path)` — `.predict(texts: list[str]) -> list[dict]`, 각 dict는 `{"text":str,"label":int,"label_name":str,"score":float}`.
  - `split_by_sentiment(results: list[dict]) -> tuple[list[str], list[str]]` — (긍정 텍스트들, 부정 텍스트들).

- [ ] **Step 1: 실패하는 테스트 작성** (순수 로직 `split_by_sentiment`만 단위테스트, 모델 로드는 통합)

```python
# tests/test_infer.py
from src.models.infer import split_by_sentiment

def test_split_by_sentiment():
    results = [
        {"text": "great", "label": 1, "label_name": "긍정", "score": 0.9},
        {"text": "bad", "label": 0, "label_name": "부정", "score": 0.8},
        {"text": "love it", "label": 1, "label_name": "긍정", "score": 0.7},
    ]
    pos, neg = split_by_sentiment(results)
    assert pos == ["great", "love it"]
    assert neg == ["bad"]
```

- [ ] **Step 2: 실패 확인**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m pytest tests/test_infer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
# src/models/infer.py
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import LABELS, MAX_LEN

class SentimentClassifier:
    def __init__(self, model_dir):
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()

    def predict(self, texts):
        enc = self.tok(texts, truncation=True, max_length=MAX_LEN,
                       padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = self.model(**enc).logits.softmax(-1)
        out = []
        for text, p in zip(texts, probs):
            label = int(p.argmax())
            out.append({"text": text, "label": label,
                        "label_name": LABELS[label], "score": float(p[label])})
        return out

def split_by_sentiment(results):
    pos = [r["text"] for r in results if r["label"] == 1]
    neg = [r["text"] for r in results if r["label"] == 0]
    return pos, neg
```

- [ ] **Step 4: 통과 확인**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m pytest tests/test_infer.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/models/infer.py tests/test_infer.py
git commit -m "feat: unified sentiment inference interface"
```

---

## Task 8: LLM 클라이언트 + 요약

**Files:**
- Create: `src/llm/client.py`
- Create: `src/llm/summarize.py`
- Test: `tests/test_summarize.py`

**Interfaces:**
- Consumes: `src.config.{LLM_MODEL_ID,LLM_FALLBACK_ID}`, 환경변수 `HF_TOKEN`
- Produces:
  - `src.llm.client.chat(prompt: str) -> str` — LLM 호출 추상화 (HF Inference API; `LLM_BACKEND=ollama`면 로컬).
  - `src.llm.summarize.build_prompt(pos: list[str], neg: list[str]) -> str` — 순수 함수, 프롬프트 문자열 생성.
  - `src.llm.summarize.summarize(pos: list[str], neg: list[str]) -> str` — `build_prompt` + `chat` 조합.

- [ ] **Step 1: 실패하는 테스트 작성** (순수 함수 `build_prompt`만 단위테스트)

```python
# tests/test_summarize.py
from src.llm.summarize import build_prompt

def test_build_prompt_includes_reviews_and_asks_pros_cons():
    p = build_prompt(["재밌다", "그래픽 좋음"], ["버그 많음"])
    assert "재밌다" in p and "버그 많음" in p
    assert "장점" in p and "단점" in p

def test_build_prompt_handles_empty_side():
    p = build_prompt([], ["별로"])
    assert "별로" in p
    assert isinstance(p, str) and len(p) > 0
```

- [ ] **Step 2: 실패 확인**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m pytest tests/test_summarize.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: client 구현**

```python
# src/llm/client.py
import os
from src.config import LLM_MODEL_ID, LLM_FALLBACK_ID

def chat(prompt: str) -> str:
    backend = os.getenv("LLM_BACKEND", "hf")
    if backend == "ollama":
        import ollama
        r = ollama.chat(model="qwen2.5",
                        messages=[{"role": "user", "content": prompt}])
        return r["message"]["content"]
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=os.environ["HF_TOKEN"])
    for model in (LLM_MODEL_ID, LLM_FALLBACK_ID):
        try:
            r = client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512)
            return r.choices[0].message.content
        except Exception:
            continue
    raise RuntimeError("모든 LLM 모델 호출 실패")
```

- [ ] **Step 4: summarize 구현**

```python
# src/llm/summarize.py
from src.llm.client import chat

def build_prompt(pos, neg):
    pos_block = "\n".join(f"- {t}" for t in pos[:20]) or "(없음)"
    neg_block = "\n".join(f"- {t}" for t in neg[:20]) or "(없음)"
    return (
        "너는 게임 리뷰 분석가다. 아래 긍정/부정 리뷰를 근거로 "
        "이 게임의 장점과 단점을 한국어로 각각 3개씩 요약하라.\n\n"
        f"[긍정 리뷰]\n{pos_block}\n\n[부정 리뷰]\n{neg_block}\n\n"
        "출력 형식:\n장점:\n- ...\n단점:\n- ...")

def summarize(pos, neg):
    return chat(build_prompt(pos, neg))
```

- [ ] **Step 5: 통과 확인**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m pytest tests/test_summarize.py -v`
Expected: PASS

- [ ] **Step 6: 실제 호출 스모크** (`.env`에 `HF_TOKEN` 설정 후)

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -c "from dotenv import load_dotenv; load_dotenv(); from src.llm.summarize import summarize; print(summarize(['재밌다'], ['버그 많음']))"`
Expected: 장점/단점 요약 텍스트 출력

- [ ] **Step 7: 커밋**

```bash
git add src/llm/ tests/test_summarize.py
git commit -m "feat: LLM client and pros/cons summarization"
```

---

## Task 9: Gradio 앱 — 탭1 (Phase 1 완성본)

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `src.models.infer.{SentimentClassifier,split_by_sentiment}`, `src.llm.summarize.summarize`
- Produces: 실행 가능한 Gradio 앱 (탭1: 감성분석+요약)

- [ ] **Step 1: app.py 작성**

```python
# app.py
import gradio as gr
from dotenv import load_dotenv
from src.config import MODEL_DIR
from src.models.infer import SentimentClassifier, split_by_sentiment
from src.llm.summarize import summarize

load_dotenv()
clf = SentimentClassifier(MODEL_DIR / "distilbert")

def analyze(reviews_text):
    reviews = [r.strip() for r in reviews_text.split("\n") if r.strip()]
    if not reviews:
        return "리뷰를 입력하세요.", ""
    results = clf.predict(reviews)
    pos, neg = split_by_sentiment(results)
    dist = f"긍정 {len(pos)}건 / 부정 {len(neg)}건"
    return dist, summarize(pos, neg)

with gr.Blocks(title="review-check") as demo:
    gr.Markdown("# review-check — Steam 리뷰 감성분석 + AI 요약")
    with gr.Tab("감성분석 + 요약"):
        inp = gr.Textbox(lines=10, label="리뷰 (한 줄에 하나)")
        btn = gr.Button("분석")
        dist = gr.Textbox(label="감성 분포")
        summary = gr.Markdown(label="AI 요약")
        btn.click(analyze, inp, [dist, summary])

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: 로컬 실행 확인**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python app.py`
Expected: `Running on local URL: http://127.0.0.1:7860` — 브라우저에서 리뷰 입력→분포+요약 확인 후 Ctrl+C

- [ ] **Step 3: 커밋**

```bash
git add app.py
git commit -m "feat: Gradio app tab1 — sentiment analysis and summary"
```

---

## Task 10: RAG 인덱스 구축 (Phase 2)

**Files:**
- Create: `src/rag/index.py`
- Create: `scripts/build_index.py`

**Interfaces:**
- Consumes: `data/train.csv`, `src.config.{EMBED_MODEL_ID,VECTOR_DIR}`
- Produces:
  - `src.rag.index.build_index(texts: list[str], persist_dir: str) -> None` — 임베딩 후 Chroma에 저장.
  - `src.rag.index.get_collection(persist_dir: str)` — 저장된 컬렉션 로드.

- [ ] **Step 1: index 구현**

```python
# src/rag/index.py
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL_ID

_embedder = None
def _embed(texts):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_ID)
    return _embedder.encode(list(texts)).tolist()

def build_index(texts, persist_dir):
    client = chromadb.PersistentClient(path=str(persist_dir))
    col = client.get_or_create_collection("reviews")
    embs = _embed(texts)
    col.add(ids=[str(i) for i in range(len(texts))],
            documents=list(texts), embeddings=embs)

def get_collection(persist_dir):
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection("reviews")
```

- [ ] **Step 2: build 스크립트 작성**

```python
# scripts/build_index.py
import pandas as pd
from src.config import DATA_DIR, VECTOR_DIR
from src.rag.index import build_index

df = pd.read_csv(DATA_DIR/"train.csv").sample(2000, random_state=42)
build_index(df["text"].tolist(), VECTOR_DIR)
print(f"indexed {len(df)} reviews into {VECTOR_DIR}")
```

- [ ] **Step 3: 실행**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m scripts.build_index`
Expected: `indexed 2000 reviews into ...`

- [ ] **Step 4: 커밋**

```bash
git add src/rag/index.py scripts/build_index.py
git commit -m "feat: RAG review embedding index"
```

---

## Task 11: RAG Q&A (Phase 2)

**Files:**
- Create: `src/rag/qa.py`
- Test: `tests/test_qa.py`

**Interfaces:**
- Consumes: `src.rag.index.get_collection`, `src.llm.client.chat`, `src.config.{VECTOR_DIR,RAG_TOP_K}`
- Produces:
  - `build_qa_prompt(question: str, contexts: list[str]) -> str` — 순수 함수.
  - `answer(question: str) -> tuple[str, list[str]]` — (답변, 근거 리뷰들).

- [ ] **Step 1: 실패하는 테스트 작성** (순수 함수만)

```python
# tests/test_qa.py
from src.rag.qa import build_qa_prompt

def test_build_qa_prompt_includes_question_and_contexts():
    p = build_qa_prompt("멀미 있어?", ["3D 멀미가 심함", "쾌적함"])
    assert "멀미 있어?" in p
    assert "3D 멀미가 심함" in p
    assert "근거" in p or "리뷰" in p
```

- [ ] **Step 2: 실패 확인**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m pytest tests/test_qa.py -v`
Expected: FAIL

- [ ] **Step 3: 구현**

```python
# src/rag/qa.py
from src.rag.index import get_collection
from src.llm.client import chat
from src.config import VECTOR_DIR, RAG_TOP_K

def build_qa_prompt(question, contexts):
    ctx = "\n".join(f"- {c}" for c in contexts)
    return (
        "아래 게임 리뷰들만 근거로 질문에 한국어로 답하라. "
        "근거가 없으면 모른다고 답하라.\n\n"
        f"[리뷰]\n{ctx}\n\n[질문] {question}\n[답변]")

def answer(question):
    col = get_collection(VECTOR_DIR)
    res = col.query(query_texts=[question], n_results=RAG_TOP_K)
    contexts = res["documents"][0]
    return chat(build_qa_prompt(question, contexts)), contexts
```

- [ ] **Step 4: 통과 확인**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python -m pytest tests/test_qa.py -v`
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add src/rag/qa.py tests/test_qa.py
git commit -m "feat: RAG question answering over reviews"
```

---

## Task 12: Gradio 앱 — 탭2 추가 (Phase 2)

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `src.rag.qa.answer`

- [ ] **Step 1: app.py에 탭2 추가** (기존 `with gr.Tab("감성분석 + 요약"):` 블록 아래, `if __name__` 위에 삽입)

```python
    with gr.Tab("리뷰 Q&A (RAG)"):
        q = gr.Textbox(label="질문", placeholder="예: 이 게임 멀미 있어?")
        qbtn = gr.Button("질문")
        ans = gr.Markdown(label="답변")
        srcs = gr.JSON(label="근거 리뷰")
        def qa(question):
            from src.rag.qa import answer
            a, ctx = answer(question)
            return a, ctx
        qbtn.click(qa, q, [ans, srcs])
```

- [ ] **Step 2: 실행 확인**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/python app.py`
Expected: 탭2에서 질문→답변+근거 리뷰 표시 확인 후 Ctrl+C

- [ ] **Step 3: 커밋**

```bash
git add app.py
git commit -m "feat: Gradio app tab2 — RAG Q&A"
```

---

## Task 13: HF Spaces 배포

**Files:**
- Create: `README.md` 상단 HF Spaces 메타데이터 (또는 별도 Space 리포)
- Modify: `requirements.txt` (Spaces용 확인)

**Interfaces:** 없음 (배포)

- [ ] **Step 1: 모델을 HF Model Hub에 업로드**

Run:
```bash
/Users/gomuseo/Desktop/Python/.venv/bin/huggingface-cli login
/Users/gomuseo/Desktop/Python/.venv/bin/python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; m=AutoModelForSequenceClassification.from_pretrained('models/distilbert'); t=AutoTokenizer.from_pretrained('models/distilbert'); m.push_to_hub('rhantj/review-check-distilbert'); t.push_to_hub('rhantj/review-check-distilbert')"
```
Expected: Hub URL 출력. 이후 `app.py`의 `SentimentClassifier`가 이 Hub ID를 로드하도록 `MODEL_DIR/"distilbert"` → `"rhantj/review-check-distilbert"`로 교체.

- [ ] **Step 2: HF Space 생성 (Gradio SDK)**

Run: `/Users/gomuseo/Desktop/Python/.venv/bin/huggingface-cli repo create review-check --type space --space_sdk gradio`
Expected: Space URL 출력

- [ ] **Step 3: Space에 HF_TOKEN 시크릿 등록**

HF Space Settings → Variables and secrets → `HF_TOKEN` 추가 (Inference API용).

- [ ] **Step 4: 코드 푸시**

```bash
git remote add space https://huggingface.co/spaces/rhantj/review-check
git push space master:main
```
Expected: Space 빌드 시작 → 앱 라이브

- [ ] **Step 5: 배포 스모크 확인**

Space URL 접속 → 탭1 리뷰 분석, 탭2 질문 동작 확인.

- [ ] **Step 6: 커밋**

```bash
git add README.md requirements.txt
git commit -m "chore: HF Spaces deployment config"
git push origin master
```

---

## Self-Review 결과

- **Spec coverage**: 데이터(Task 2-3), DL 2종 비교(Task 4-6), 통합추론(7), LLM 요약(8), Gradio 탭1(9), RAG(10-11), 탭2(12), 배포(13) — 스펙 전 항목 매핑됨.
- **Placeholder**: Task 3 Step 2에서 데이터셋 ID를 런타임 확인 후 확정하도록 명시(플레이스홀더 아님, 검증 절차).
- **Type 일관성**: `predict()` 반환 dict 키(`text/label/label_name/score`), `split_by_sentiment`/`summarize`/`answer` 시그니처가 Task 간 일치.
