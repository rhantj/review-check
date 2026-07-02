from src.llm.client import chat

def build_prompt(pos, neg):
    pos_block = "\n".join(f"- {t}" for t in pos[:20]) or "(없음)"
    neg_block = "\n".join(f"- {t}" for t in neg[:20]) or "(없음)"
    return (
        "너는 게임 리뷰 분석가다. 아래 긍정/부정 리뷰를 근거로 "
        "이 게임의 장점과 단점을 한국어로 각각 3개씩 요약하라.\n\n"
        f"[긍정 리뷰]\n{pos_block}\n\n[부정 리뷰]\n{neg_block}\n\n"
        "출력 형식:\n장점:\n- ...\n단점:\n- ...")

def summarize(pos, neg):
    return chat(build_prompt(pos, neg))
