# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Steam 게임 리뷰 감성분석(LSTM vs DistilBERT 비교) + LLM(Qwen3-Instruct) 요약 + RAG Q&A를 제공하는 Streamlit 데모. 포트폴리오/발표용 프로젝트.

## Commands

```bash
pip install -r requirements.txt          # 의존성 설치
cp .env.example .env                     # HF_TOKEN 설정 필요
streamlit run app.py                     # 앱 실행
```

빌드/린트 설정은 없음. `pytest`가 requirements에 있지만 `tests/`는 비어 있다 — 아래 "노트북·테스트 실행 정책" 참고.

## 노트북·테스트 실행 정책 (중요)

- **모델 학습·실험 노트북 실행은 사용자가 직접 진행한다.** Claude는 노트북 코드 작성까지만 하고, 실행(`jupyter nbconvert --execute` 등)은 사용자가 명시적으로 요청할 때만 한다.
- **실행 절차성 코드(데이터 다운로드/전처리/학습/비교/인덱스 구축)는 `.py` 스크립트가 아닌 `notebooks/*.ipynb`로 작성한다.** `app.py`와 `src/` 모듈은 HF Spaces 배포에 필요하므로 `.py`로 유지.
- **`tests/` 파일은 만들지 않는다.** 테스트 작성은 사용자가 직접 한다.
- 그래프·리포트 산출물은 각 노트북에서 `output/`(`src.config.OUTPUT_DIR`)에 `fig.savefig(...)`로 저장한다 (파일명 접두사는 노트북 번호, 예: `03_lstm_loss_confusion.png`). `output/*.png`는 `.gitignore` 대상이며 폴더 자체는 `.gitkeep`으로 유지.

## 노트북 실행 순서

노트북은 서로 파일시스템으로 의존하므로(다음 노트북이 이전 산출물을 읽음) **순서대로만** 실행해야 한다:

1. `01_download_data.ipynb` — HF `ksang/steamreviews`에서 긍/부정 각 1만 건 균형 샘플링 → `data/raw_sample.csv` + 원본 EDA
2. `02_preprocessing.ipynb` — `src.data.pipeline` 정제(BBCode/URL 제거, 반복문자 축약, 구두점 분리, 최소 길이 필터) → 전후 비교 → `data/{train,val,test}.csv` (계층 분할 70/15/15)
3. `03_train_lstm.ipynb` — LSTM 베이스라인 학습 → `models/lstm/{model.pt,vocab.json,metrics.json}`
4. `04_train_distilbert.ipynb` — `distilbert-base-uncased` 파인튜닝 → `models/distilbert/`(모델+토크나이저+`metrics.json`)
5. `05_compare_models.ipynb` — 두 모델 `metrics.json`을 읽어 비교표 생성 → `models/comparison.md`
6. `06_build_index.ipynb` — train 2,000건 샘플 임베딩(`all-MiniLM-L6-v2`) → `chroma_store/`에 색인
7. `07_model_analysis.ipynb` — DistilBERT 대상 ROC/PR, 확신도 분포, 오분류 사례, gradient saliency, attention 히트맵 (04 완료 필요)

`app.py`는 `models/distilbert/`(4번 산출물)와 `chroma_store/`(6번 산출물)에 의존한다.

## Architecture

5개 컴포넌트가 파일시스템(`data/`, `models/`, `chroma_store/`)을 매개로 느슨하게 연결된 파이프라인이다. 코드는 실행 로직(노트북)과 재사용 로직(`src/`)으로 분리되어 있고, 노트북은 `src/`를 import해서 쓴다.

```
src/config.py         전역 상수(경로, 모델 ID, RANDOM_SEED, LABELS) — 다른 모든 모듈이 여기서 경로를 가져옴
src/data/pipeline.py  텍스트 정제(normalize_text) + 정제·라벨변환(clean_reviews) + 계층분할(split_data)
src/models/dataset.py LSTM용 어휘 구축(build_vocab)·인코딩(encode) — 공백 토크나이저, PAD=0/UNK=1
src/models/lstm.py    LSTMClassifier (nn.Module)
src/models/infer.py   SentimentClassifier — HF AutoModel 기반 추론 래퍼, app.py가 사용하는 것은 이 클래스
src/llm/client.py     chat() — LLM_BACKEND 환경변수로 HF Inference API(기본)/Ollama 전환, 1차 모델 실패 시 LLM_FALLBACK_ID로 폴백
src/llm/summarize.py  긍정/부정 리뷰 목록 → 장단점 요약 프롬프트 + 호출
src/rag/index.py      SentenceTransformer 임베딩 → Chroma PersistentClient 색인/조회
src/rag/qa.py         질문 → 벡터 검색(top-k) → LLM 근거 기반 답변
```

`app.py`는 Streamlit 2탭 구조: 탭1(`SentimentClassifier` + `summarize`), 탭2(`src.rag.qa.answer`, RAG 모듈은 지연 import).

LLM 호출은 `src/llm/client.py` 한 곳으로 추상화되어 있어 로컬 Ollama 테스트와 배포용 HF Inference API 전환이 `LLM_BACKEND` 환경변수 하나로 이뤄진다.

라벨 규약: Steam `review_score`는 1(추천)/-1(비추천)이며 `clean_reviews`에서 1(긍정)/0(부정)으로 변환된다. `src.config.LABELS`가 이 매핑을 표시용으로 다시 정의한다(`{0: "부정", 1: "긍정"}`).

`docs/specs/`와 `docs/plans/`에 초기 설계 스펙과 구현 계획이 있다 — 아키텍처 배경(왜 LSTM vs DistilBERT를 비교하는지, Phase 1/2 스코프 등)이 필요하면 참고.

표, 그래프 등의 산출물들은 반드시 output 폴더에 저장한다. 코드 수정 후 변경된 산출물들은 네이밍을 달리해서 저장한다.
