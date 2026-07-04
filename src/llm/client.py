import os
from src.config import LLM_MODEL_ID, LLM_FALLBACK_ID

def chat(prompt: str) -> str:
    backend = os.getenv("LLM_BACKEND", "hf")
    if backend == "ollama":
        import ollama
        r = ollama.chat(model="qwen2.5",
                        messages=[{"role": "user", "content": prompt}])
        return r["message"]["content"]
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=os.environ["HF_TOKEN"])
    errors = []
    for model in (LLM_MODEL_ID, LLM_FALLBACK_ID):
        try:
            r = client.chat_completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512)
            return r.choices[0].message.content
        except Exception as e:
            errors.append(f"{model}: {type(e).__name__} {e}")
    raise RuntimeError("모든 LLM 모델 호출 실패:\n" + "\n".join(errors))
