from pathlib import Path

RANDOM_SEED = 42
MAX_LEN = 128  # 추론 토큰 상한 — 학습(TRAIN_MAX_LEN=128)과 일치, 초과분은 낭비

HUB_MODEL_ID = "rhantj/review-check-distilbert"  # 로컬 models/distilbert 부재 시 폴백
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LLM_FALLBACK_ID = "meta-llama/Llama-3.3-70B-Instruct"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K = 5

LABELS = {0: "부정", 1: "긍정"}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
VECTOR_DIR = ROOT / "chroma_store"
OUTPUT_DIR = ROOT / "output"
