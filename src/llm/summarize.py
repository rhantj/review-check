from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.llm.client import get_chat_model

# 프롬프트 템플릿: {ratio}/{pos_block}/{neg_block} 자리에 값이 채워진다
_PROMPT = ChatPromptTemplate.from_template(
    "너는 게임 리뷰 분석가다. 아래 긍정/부정 리뷰와 비율을 근거로 "
    "이 게임을 한국어로 요약하라. 총평은 긍/부정 비율을 반영해 "
    "여론의 방향을 2~3문장으로 쓰고, 장점과 단점은 각각 3개씩 쓴다.\n\n"
    "[긍/부정 비율] {ratio}\n\n"
    "[긍정 리뷰]\n{pos_block}\n\n[부정 리뷰]\n{neg_block}\n\n"
    "출력 형식:\n총평: ...\n\n장점:\n- ...\n단점:\n- ...")


def summarize(pos, neg):
    total = len(pos) + len(neg)
    ratio = (f"전체 {total}건 중 긍정 {len(pos)}건({len(pos) / total:.0%}), "
             f"부정 {len(neg)}건({len(neg) / total:.0%})" if total else "정보 없음")

    # 체인: 프롬프트 → LLM → 문자열 파싱
    chain = _PROMPT | get_chat_model() | StrOutputParser()
    return chain.invoke({
        "ratio": ratio,
        "pos_block": "\n".join(f"- {t}" for t in pos[:20]) or "(없음)",
        "neg_block": "\n".join(f"- {t}" for t in neg[:20]) or "(없음)",
    })
