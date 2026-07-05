# review-check

Steam 게임 리뷰 감성분석 + AI 요약 + RAG Q&A

직접 학습한 딥러닝 감성분류 모델과 LLM(Gemma-3-27B)을 결합해, 게임 리뷰의 긍/부정을 분류하고 장단점을 자연어로 요약하며 리뷰 기반 질의응답을 제공하는 웹 데모.

## 구성

| 컴포넌트 | 내용 |
|---|---|
| 데이터 | Steam Reviews (HF `ksang/steamreviews`, 긍/부정 1:1 균형 1만 건 샘플) |
| DL 감성분류 | LSTM 직접학습 vs DistilBERT 파인튜닝 (비교) |
| LLM 요약/답변 | google/gemma-3-27b-it (HF Inference API, 폴백 Qwen2.5-7B) |
| RAG | LangChain (all-MiniLM-L6-v2 임베딩 + Chroma 검색 + LCEL 체인) |
| UI/배포 | Streamlit 3탭 (게임 분석 / 직접 입력 / RAG Q&A) → Streamlit Cloud |

## 실행 절차

1. 의존성 설치: `pip install -r requirements.txt`
2. 프로젝트 루트에 `.env` 파일을 만들고 `HF_TOKEN=hf_...` 설정
3. 노트북을 순서대로 실행해 데이터·모델·인덱스 생성:
   - `notebooks/01_download_data.ipynb` — 데이터 다운로드 + 원본 EDA (`data/raw_sample.csv`)
   - `notebooks/02_preprocessing.ipynb` — 전처리 전후 비교 + 정제·분할 (`data/{train,val,test}.csv`)
   - `notebooks/03_train_lstm.ipynb` — LSTM + TF-IDF 베이스라인 학습 (`models/lstm/`, `models/tfidf_logreg/`)
   - `notebooks/04_train_distilbert.ipynb` — DistilBERT 파인튜닝 (`models/distilbert/`)
   - `notebooks/05_compare_models.ipynb` — 세 모델 지표 비교표
   - `notebooks/06_build_index.ipynb` — RAG 벡터 인덱스 구축 (`chroma_store/`)
   - `notebooks/07_model_analysis.ipynb` — 모델 해석 (ROC/PR, 오분류, saliency, attention)
4. 앱 실행: `streamlit run app.py`

## 프로젝트 구조

```
app.py            Streamlit 2탭 앱 (감성분석+요약 / RAG Q&A)
src/              라이브러리 모듈 (config, 데이터 파이프라인, 모델, LLM, RAG)
notebooks/        데이터 준비·학습·비교·인덱스 구축 노트북
docs/             설계 스펙과 구현 계획
```
