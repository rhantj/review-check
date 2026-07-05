import os

from src.config import LLM_MODEL_ID, LLM_FALLBACK_ID


def get_chat_model():
    """LangChain 챗 모델을 반환한다.

    - LLM_BACKEND=ollama : 로컬 Ollama (테스트용)
    - 기본(hf)           : HF Inference API — 1차 모델 실패 시 폴백 모델로 자동 전환
    """
    if os.getenv("LLM_BACKEND", "hf") == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model="qwen2.5")

    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    def hf_chat(model_id):
        endpoint = HuggingFaceEndpoint(
            repo_id=model_id, task="conversational",
            huggingfacehub_api_token=os.environ["HF_TOKEN"],
            max_new_tokens=512)
        return ChatHuggingFace(llm=endpoint)

    return hf_chat(LLM_MODEL_ID).with_fallbacks([hf_chat(LLM_FALLBACK_ID)])


def chat(prompt: str) -> str:
    """문자열 프롬프트 → 문자열 응답 헬퍼."""
    return get_chat_model().invoke(prompt).content
