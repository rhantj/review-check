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
