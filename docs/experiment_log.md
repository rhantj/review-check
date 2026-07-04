# 실험 로그 — review-check

Steam 리뷰 감성분석 (LSTM vs DistilBERT) + LLM 요약 + RAG Q&A 프로젝트의 실험·변경 이력.
날짜는 작업일 기준. 상세 수치·그림은 `output/`과 `docs/report_slides.html` 참고.

---

## 1. 데이터셋 선정 (2026-07-02)

- 구현 계획의 후보 데이터셋들이 HF Hub에 없거나 로드 실패 → HF API 탐색으로 **`ksang/steamreviews`** (Kaggle Steam Reviews 미러, 641만 건, `review_text` + `review_score` 1/-1) 확정.
- 앞부분만 자르면 app_id 순 정렬 때문에 특정 게임에 편중 → **셔플 후 샘플링**.

## 2. 초기 파이프라인 구축 (2026-07-02 ~ 07-03)

- 2만 건 샘플(자연 분포 82:18) → 기본 정제 → 70/15/15 계층 분할.
- **LSTM (3에폭)**: acc 0.819 / F1 0.900 — 이후 다수 클래스 붕괴로 판명 (아래 6번).
- **DistilBERT 파인튜닝**: acc 0.903 / F1 0.941.
  - 시행착오: ① unpadded 시퀀스 stack 실패 → `DataCollatorWithPadding` 추가, ② 16GB MPS 커널 사망(silent OOM) → batch 16→8 + gradient accumulation 2, ③ 명시적 MPS OOM → max_len 256→128.
- RAG: train 2,000건 `all-MiniLM-L6-v2` 임베딩 → Chroma 색인. Streamlit 2탭 앱 작성.

## 3. 전처리 확장 (2026-07-03)

- 01 노트북에 결측치·노이즈 진단 추가. 실측: **"Early Access Review" 보일러플레이트 15.8%**, HTML 엔티티 2.0%, BBCode 5%대, URL, 3+ 반복문자, 기호-only 0.4%, 3단어 미만.
- `normalize_text` 확장: HTML unescape, Early Access 접두 문구 제거, BBCode/URL 제거, 반복문자 축약, 구두점 분리. `clean_reviews`에 기호-only 행 제거, `min_words=3` 필터.
- 효과: 어휘 크기 27,014 → 19,864 (-27%) — LSTM 공백 토크나이저의 어휘 낭비 감소.

## 4. 판별 토큰 분석 — 3번의 통계 왜곡 수정 (2026-07-03 ~ 07-04)

단순 빈도 상위 토큰은 "game" 같은 도메인 최빈어만 나와 정보가 없어 **로그오즈비(log-odds, α=1 스무딩)** 로 교체. 이후 실데이터에서 왜곡 3건을 발견·수정:

| 발견 | 원인 | 수정 |
|---|---|---|
| ( ͡° ͜ʖ ͡°) 밈 조각이 긍정 상위 + AppleGothic 글리프 경고 | 밈 복사붙여넣기 스팸 | 비ASCII 토큰 제외 |
| `>` (그린텍스트 인용 기호) | 밈 문체 | 영숫자 없는 토큰 제외 |
| `salad`가 부정 판별 토큰 | 부정 리뷰 1건(1,296단어)이 48회 반복 | **문서 빈도**(리뷰당 1회) 집계로 전환 |

- 잔여 한계: 특정 게임 캐릭터명(katara 등) 혼입 = 샘플링 편향. app_id 층화 샘플링으로 완화 가능(미적용).

## 5. 균형 샘플링 전환 (2026-07-03)

- 원본 82:18 불균형이 LSTM 다수 클래스 붕괴의 원인 → **클래스별 셔플 후 50:50 균형 샘플**로 전환.
- 학습 시간 절충으로 2만 → **1만 건**(긍/부정 각 5천). 주의: 균형 test의 accuracy는 실제 분포 성능과 다름 (보고서에 명시).

## 6. LSTM 패딩 버그 발견·해결 (2026-07-04)

- 균형 데이터 전환 후에도 train loss가 0.693(=ln 2, 무작위 수준)에 고정, test acc 0.510.
- **원인**: 모든 리뷰를 SEQ_MAX_LEN까지 PAD(0)로 채우는데, forward가 패딩까지 전부 처리한 마지막 hidden state 사용 → 짧은 리뷰는 수백 스텝의 0벡터가 내용 신호를 씻어냄.
- **수정**: `pack_padded_sequence`로 실제 길이까지만 LSTM 처리 (`src/models/lstm.py`).
- **결과**: test acc 0.510 → **0.763** / F1 0.753. 혼동행렬도 균형적으로 개선.
- 교훈 ①: 불균형 데이터에서 acc 0.819는 "전부 긍정 찍기"였고, 균형 샘플링이 숨은 버그를 노출했다.
- 교훈 ②: `src/` 수정 후 커널 재시작 없이 재실행하면 옛 모듈이 캐시됨 — 재실행 전 **커널 재시작 필수** (같은 원인의 헛실험 2회).

## 7. 학습 프로토콜 정비 (2026-07-04)

- **LSTM**: 고정 3에폭 → 최대 에폭 + **val loss 기준 early stopping** + 최적 시점 가중치 복원. 하이퍼파라미터 전용 셀 분리 (MAX_EPOCHS, PATIENCE, BATCH_SIZE, LR, SEQ_MAX_LEN, EMBED_DIM, HIDDEN, MIN_FREQ).
  - 사용자 튜닝 결과: MAX_EPOCHS=100, PATIENCE=10, BATCH=128, LR=1e-4, SEQ_MAX_LEN=512 → best epoch 21, val loss 0.5013.
- **DistilBERT**: 2→3에폭 + `load_best_model_at_end`(val F1 기준) + 체크포인트 자동 정리. 하이퍼파라미터 셀 분리 (EPOCHS, LR, WEIGHT_DECAY, TRAIN_MAX_LEN, BATCH_SIZE, GRAD_ACCUM).
- 옵티마이저는 Adam 유지 (이 규모의 표준, 대조군 과튜닝 방지).

**에폭 수가 두 모델에서 반대인 이유** (발표용 논거):

| | LSTM (21에폭 필요) | DistilBERT (3에폭 충분) |
|---|---|---|
| 출발점 | 무작위 초기화 — 임베딩부터 전부 학습 | 사전학습 완료 — 분류 헤드(전체의 ~1%)만 신규 |
| 배울 양 | 많음 → 수렴 느림 | 적음 → ~1,000 스텝(3에폭)이면 수렴 |
| 과적합 속도 | 파라미터 적어 암기 느림 → 오래 돌려도 비교적 안전 | 66M 파라미터 vs 5.7k 샘플 → 몇 에폭 만에 암기 시작, 사전학습 지식 손상(catastrophic forgetting) 위험 |
| 대응 | early stopping으로 최적 시점 탐색 | 2~4에폭(BERT 논문 권장) + val 최적 에폭 자동 선택 |

## 8. TF-IDF + 로지스틱 회귀 베이스라인 추가 (2026-07-04)

- 비교 구도를 **고전 ML → 비사전학습 DL → 사전학습 DL** 3단으로 확장 (03 노트북, 05 비교표 갱신).
- 결과: **acc 0.818 / F1 0.813 — LSTM(0.763)보다 우세.** "딥러닝 ≠ 무조건 우월, 관건은 사전학습"의 정량 근거.
- 부가 산출물: 로지스틱 회귀 계수 상위 차트(`03_tfidf_logreg_coefs.png`) — 02 로그오즈비와 교차 확인용.

## 9. 학습 곡선 실험 (2026-07-04)

- train 1k/2k/4k/전체 슬라이스별 LSTM·TF-IDF 학습 → test acc 곡선 + DistilBERT 참고선 (`03_learning_curve.{png,csv}`).
- 결과: **가설대로 LSTM이 데이터 증가의 수혜가 가장 큼** — 1k→5.7k에서 LSTM +11.8%p(0.638→0.756), TF-IDF +4.9%p(0.769→0.818). LSTM 곡선은 전체 구간에서 아직 상승 중(데이터 부족), TF-IDF는 완만. DistilBERT(0.856)는 모든 지점 위 — "사전학습 = 데이터를 공짜로 얻는 효과"의 실증.

## 10. LLM 전환 (2026-07-03)

- Qwen2.5-7B-Instruct → **Qwen3-30B-A3B-Instruct-2507** (HF Inference API 서빙 확인).
- 선정 이유: non-thinking 전용이라 `<think>` 블록이 출력에 안 섞임, MoE라 빠름, 한국어 품질 개선. 추론은 HF 서버에서 수행되어 로컬 자원과 무관 (크레딧·지연만 영향).
- 폴백: `google/gemma-2-9b-it` 유지.

---

## 현재 성능 요약 (균형 1만 건, 동일 분할, 최종)

| 모델 | Accuracy | F1 | 비고 |
|---|---|---|---|
| LSTM | 0.763 | 0.753 | early stopping, best epoch 21 |
| TF-IDF+LogReg | 0.818 | 0.813 | 학습 수 초 — LSTM보다 우세 |
| **DistilBERT** | **0.856** | **0.855** | 3에폭, val F1 최적 에폭 선택 |

3단 순위(고전 ML > 비사전학습 DL, 사전학습 DL 최종 우위)가 "딥러닝의 우위는 아키텍처가 아니라 사전학습에서 나온다"를 보여줌. 균형 test(기저율 50%) 기준이므로 이전 불균형 실험 수치(DistilBERT 0.903)와 직접 비교 불가.

## 남은 작업

- [x] 04 DistilBERT 균형 1만 건 재학습 (acc 0.856) → 05 비교표·보고서 반영 완료
- [x] 03 학습 곡선 실행 → 보고서 반영 완료
- [x] 06 RAG 인덱스 재구축
- [x] 07 모델 해석 실행 → 보고서 반영 완료 (ROC-AUC 0.929 / PR-AUC 0.930, 오분류 176/1,219 — 확신 오분류 top-5 전부 "혼합 감성 부정 리뷰를 긍정으로" 유형. transformers 5.x SDPA가 output_attentions 미지원 → attention 셀만 eager 구현으로 재로드해 해결)
- [ ] `streamlit run app.py` 동작 확인 (`.env` HF_TOKEN 설정 완료)
- [ ] HF Model Hub 업로드 + Spaces 배포 (Task 13)
