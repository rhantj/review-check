from pathlib import Path

RANDOM_SEED = 42
MAX_LEN = 256

HUB_MODEL_ID = "rhantj/review-check-distilbert"  # 로컬 models/distilbert 부재 시 폴백
LLM_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LLM_FALLBACK_ID = "google/gemma-2-9b-it"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K = 5

LABELS = {0: "부정", 1: "긍정"}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
VECTOR_DIR = ROOT / "chroma_store"
OUTPUT_DIR = ROOT / "output"
