# review-check

Steam 게임 리뷰 감성분석 + AI 요약 + RAG Q&A

직접 학습한 딥러닝 감성분류 모델과 LLM(Qwen2.5-Instruct)을 결합해, 게임 리뷰의 긍/부정을 분류하고 장단점을 자연어로 요약하며 리뷰 기반 질의응답을 제공하는 웹 데모.

## 구성

| 컴포넌트 | 내용 |
|---|---|
| 데이터 | Steam Reviews (Kaggle) |
| DL 감성분류 | LSTM/GRU 직접학습 vs DistilBERT 파인튜닝 (비교) |
| LLM 요약/답변 | Qwen2.5-Instruct (HF Inference API) |
| RAG | Chroma/FAISS 리뷰 검색 |
| UI/배포 | Gradio 2탭 → Hugging Face Spaces |

## 상태

설계 단계. 상세 스펙은 `docs/specs/` 참고.
