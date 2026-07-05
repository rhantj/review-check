from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import EMBED_MODEL_ID

_embeddings = None


def get_embeddings():
    """임베딩 모델 (최초 1회만 로드)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    return _embeddings


def get_vectorstore(persist_dir):
    """디스크에 영속화된 Chroma 벡터스토어."""
    return Chroma(collection_name="reviews",
                  persist_directory=str(persist_dir),
                  embedding_function=get_embeddings())


def build_index(texts, persist_dir, metadatas=None, batch_size=2000):
    vs = get_vectorstore(persist_dir)
    vs.reset_collection()  # 재구축 시 기존 색인 비우고 새로 시작
    texts = list(texts)
    # Chroma의 1회 add 배치 상한을 넘지 않도록 나눠서 추가
    for s in range(0, len(texts), batch_size):
        e = s + batch_size
        vs.add_texts(texts[s:e],
                     metadatas=metadatas[s:e] if metadatas else None,
                     ids=[str(i) for i in range(s, min(e, len(texts)))])


def get_collection(persist_dir):
    """(호환용 별칭) 메타데이터 조회(.get)를 지원하는 벡터스토어 반환."""
    return get_vectorstore(persist_dir)
