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
