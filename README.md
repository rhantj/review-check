# review-check

Steam 게임 리뷰 감성분석 + AI 요약 + RAG Q&A

직접 학습한 딥러닝 감성분류 모델과 LLM(Qwen2.5-Instruct)을 결합해, 게임 리뷰의 긍/부정을 분류하고 장단점을 자연어로 요약하며 리뷰 기반 질의응답을 제공하는 웹 데모.

## 구성

| 컴포넌트 | 내용 |
|---|---|
| 데이터 | Steam Reviews (HF `ksang/steamreviews`, 2만 건 샘플) |
| DL 감성분류 | LSTM 직접학습 vs DistilBERT 파인튜닝 (비교) |
| LLM 요약/답변 | Qwen2.5-Instruct (HF Inference API) |
| RAG | all-MiniLM-L6-v2 임베딩 + Chroma 리뷰 검색 |
| UI/배포 | Streamlit 2탭 → Hugging Face Spaces |

## 실행 절차

1. 의존성 설치: `pip install -r requirements.txt`
2. `.env.example`을 `.env`로 복사하고 `HF_TOKEN` 설정
3. 노트북을 순서대로 실행해 데이터·모델·인덱스 생성:
   - `notebooks/01_download_data.ipynb` — 데이터 다운로드·정제·분할 (`data/`)
   - `notebooks/02_train_lstm.ipynb` — LSTM 베이스라인 학습 (`models/lstm/`)
   - `notebooks/03_train_distilbert.ipynb` — DistilBERT 파인튜닝 (`models/distilbert/`)
   - `notebooks/04_compare_models.ipynb` — 두 모델 지표 비교표
   - `notebooks/05_build_index.ipynb` — RAG 벡터 인덱스 구축 (`chroma_store/`)
4. 앱 실행: `streamlit run app.py`

## 프로젝트 구조

```
app.py            Streamlit 2탭 앱 (감성분석+요약 / RAG Q&A)
src/              라이브러리 모듈 (config, 데이터 파이프라인, 모델, LLM, RAG)
notebooks/        데이터 준비·학습·비교·인덱스 구축 노트북
docs/             설계 스펙과 구현 계획
```
