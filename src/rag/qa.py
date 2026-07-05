from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import VECTOR_DIR, RAG_TOP_K
from src.llm.client import get_chat_model
from src.rag.index import get_vectorstore

_PROMPT = ChatPromptTemplate.from_template(
    "아래 게임 리뷰들만 근거로 질문에 한국어로 답하라. "
    "근거가 없으면 모른다고 답하라.\n\n"
    "[리뷰]\n{context}\n\n[질문] {question}\n[답변]")


def answer(question, app_name=None):
    # ① 검색: 질문과 유사한 리뷰 top-k (게임 필터 선택)
    search_kwargs = {"k": RAG_TOP_K}
    if app_name:
        search_kwargs["filter"] = {"app_name": app_name}
    retriever = get_vectorstore(VECTOR_DIR).as_retriever(search_kwargs=search_kwargs)
    docs = retriever.invoke(question)
    contexts = [d.page_content for d in docs]

    # ② 생성: 검색된 리뷰를 근거로 넣어 LLM 호출
    chain = _PROMPT | get_chat_model() | StrOutputParser()
    ans = chain.invoke({"context": "\n".join(f"- {c}" for c in contexts),
                        "question": question})
    return ans, contexts
