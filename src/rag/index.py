import chromadb
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL_ID

_embedder = None
def _embed(texts):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_ID)
    return _embedder.encode(list(texts)).tolist()

def build_index(texts, persist_dir, metadatas=None, batch_size=2000):
    client = chromadb.PersistentClient(path=str(persist_dir))
    # 재구축 시 기존 색인과 ID가 충돌하지 않도록 컬렉션을 새로 만든다
    try:
        client.delete_collection("reviews")
    except Exception:
        pass
    col = client.create_collection("reviews")
    texts = list(texts)
    embs = _embed(texts)
    # Chroma의 1회 add 배치 상한을 넘지 않도록 나눠서 추가
    for s in range(0, len(texts), batch_size):
        e = s + batch_size
        col.add(ids=[str(i) for i in range(s, min(e, len(texts)))],
                documents=texts[s:e], embeddings=embs[s:e],
                metadatas=metadatas[s:e] if metadatas else None)

def get_collection(persist_dir):
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection("reviews")
