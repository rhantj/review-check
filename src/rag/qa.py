from src.rag.index import get_collection
from src.llm.client import chat
from src.config import VECTOR_DIR, RAG_TOP_K

def build_qa_prompt(question, contexts):
    ctx = "\n".join(f"- {c}" for c in contexts)
    return (
        "아래 게임 리뷰들만 근거로 질문에 한국어로 답하라. "
        "근거가 없으면 모른다고 답하라.\n\n"
        f"[리뷰]\n{ctx}\n\n[질문] {question}\n[답변]")

def answer(question, app_name=None):
    col = get_collection(VECTOR_DIR)
    res = col.query(query_texts=[question], n_results=RAG_TOP_K,
                    where={"app_name": app_name} if app_name else None)
    contexts = res["documents"][0]
    return chat(build_qa_prompt(question, contexts)), contexts
