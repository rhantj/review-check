# review-check

Steam 게임 리뷰 감성분석 + AI 요약 + RAG Q&A

직접 학습한 딥러닝 감성분류 모델과 LLM(Qwen2.5-Instruct)을 결합해, 게임 리뷰의 긍/부정을 분류하고 장단점을 자연어로 요약하며 리뷰 기반 질의응답을 제공하는 웹 데모.

## 구성

| 컴포넌트 | 내용 |
|---|---|
| 데이터 | Steam Reviews (HF `ksang/steamreviews`, 긍/부정 1:1 균형 1만 건 샘플) |
| DL 감성분류 | LSTM 직접학습 vs DistilBERT 파인튜닝 (비교) |
| LLM 요약/답변 | Qwen/Qwen2.5-7B-Instruct (HF Inference API, 폴백 Qwen2.5-72B-Instruct) |
| RAG | LangChain (all-MiniLM-L6-v2 임베딩 + Supabase pgvector 검색 + LCEL 체인) |
| UI/배포 | Streamlit 3탭 (게임 분석 / 직접 입력 / RAG Q&A) → Streamlit Cloud |

## RAG 구조

```
일반 LLM:   질문 ──────────────────→ LLM → 답변 (근거 없음, 환각 위험)
RAG:        질문 → [내 데이터 검색] → 근거와 함께 LLM → 답변 (근거 제시 가능)
```

| # | 부품 | 역할 | 구현 |
|---|---|---|---|
| ① | 임베딩 모델 | 문장 → 384차원 벡터 | `all-MiniLM-L6-v2` (`HuggingFaceEmbeddings`) |
| ② | 벡터 DB | 리뷰 벡터 저장 + 유사도 검색 | Supabase Postgres + pgvector (`langchain-postgres`, HNSW 인덱스) |
| ③ | 리트리버 | 질문과 유사한 리뷰 top-5 검색, 게임 필터(`app_name`) 지원 | `src/rag/qa.py` |
| ④ | 프롬프트 | 검색된 리뷰만 근거로 답하도록 지시 (근거 없으면 "모른다") | `ChatPromptTemplate` |
| ⑤ | LLM | 근거를 읽고 한국어로 답변 생성 | Qwen2.5-7B-Instruct (폴백 Qwen2.5-72B-Instruct) |

5개 부품이 LCEL 체인 한 줄로 조립된다: `prompt | llm | StrOutputParser()`.

- 색인 구축(오프라인, 1회): `notebooks/06_build_index.ipynb` — 균형 샘플링한 리뷰를 임베딩해 `app_name`/`label` 메타데이터와 함께 pgvector에 체크포인트 기반 배치 적재, 완료 후 HNSW 인덱스 생성.
- 로컬 Chroma(파일 기반)에서 Supabase pgvector로 이전한 이유: 게임별/라벨별 메타데이터 필터링을 SQL로 직접 다룰 수 있고, 전체 리뷰(수십만~수백만 건) 규모로 확장할 때 파일 기반 벡터스토어의 메모리·용량 한계(Streamlit Cloud, git 100MB 제한)를 벗어날 수 있어서.

## LLM 비교 실험

생성 모델은 세 차례 교체를 거쳐 현재 조합(Qwen2.5-7B 1차 + Qwen2.5-72B 폴백)으로 정착했다.

| 후보 | 결과 | 이유 |
|---|---|---|
| Qwen3-30B-A3B-Instruct-2507 | 기각 | HF Inference 라우터가 계정에 미제공 (`model_not_supported`) |
| Qwen3.5 / Qwen3.6 계열 | 기각 | reasoning 토큰만 소모하고 응답이 빈 문자열로 반환 |
| gemma-3-27b-it | 임시 채택 후 교체 | 라우터 실호출은 되나 한국어 출력 품질이 최종 조합보다 낮음 |
| **Qwen2.5-7B-Instruct** (1차) | **최종 채택** | non-thinking 전용이라 `<think>` 블록 미혼입, MoE로 응답 속도 빠름, 한국어 품질 양호 |
| Llama-3.3-70B-Instruct (폴백, 1차 결정) | 이후 교체 | 1차 모델과 계열이 달라 응답 톤/포맷이 흔들림 |
| **Qwen2.5-72B-Instruct** (폴백, 최종) | **최종 채택** | 1차·폴백을 Qwen2.5 계열로 통일해 응답 스타일 일관성 확보 |

- LLM·RAG 체인은 `InferenceClient` 직접 호출 방식에서 **LangChain 표준 구조**(`ChatHuggingFace` + `with_fallbacks`, LCEL `prompt | llm | parser`)로 재작성 — 폴백 전환이 코드 한 곳(`with_fallbacks`)에 캡슐화됨.
- `LLM_BACKEND` 환경변수로 배포용 HF Inference API와 로컬 개발용 Ollama(`ChatOllama`)를 전환 가능.
- 추론은 HF 서버에서 수행되므로 로컬 자원과 무관 — 크레딧·지연에만 영향.

## 감성분류 모델 비교

균형 샘플(긍/부정 1:1), 동일 test 분할 기준:

| 모델 | Accuracy | F1 | 비고 |
|---|---|---|---|
| LSTM | 0.763 | 0.753 | 무작위 초기화부터 학습, early stopping(best epoch 21) |
| TF-IDF + LogReg | 0.818 | 0.813 | 학습 수 초 — LSTM보다 우세 |
| **DistilBERT** | **0.856** | **0.855** | 사전학습 파인튜닝, 3에폭 |

고전 ML(TF-IDF)이 비사전학습 DL(LSTM)보다 우세하고, 사전학습 DL(DistilBERT)이 최종 우위 — "딥러닝의 우위는 아키텍처가 아니라 사전학습에서 나온다"는 것을 정량적으로 보여준다.

## 프로젝트 구조

```
app.py            Streamlit 3탭 앱 (게임 분석 / 직접 입력 / RAG Q&A)
src/              라이브러리 모듈 (config, 데이터 파이프라인, 모델, LLM, RAG)
notebooks/        데이터 준비·학습·비교·인덱스 구축 노트북
docs/             설계 스펙과 구현 계획
```
