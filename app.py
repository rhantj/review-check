import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
try:
    # Streamlit Cloud는 .env가 없고 st.secrets로 환경변수를 주입한다 —
    # src.config가 os.environ에서 읽으므로 여기서 연결해준다 (로컬은 secrets.toml 없어 예외 무시).
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass

from src.config import MODEL_DIR, HUB_MODEL_ID
from src.rag.index import get_game_counts as _get_game_counts, get_reviews_by_app

MIN_GAME_REVIEWS = 20  # 게임 분석 탭에 노출할 최소 리뷰 수

@st.cache_resource
def get_classifier():
    # torch/transformers는 무거우므로 실제 분석 시점에만 import (기동 메모리 절약)
    from src.models.infer import SentimentClassifier
    local = MODEL_DIR / "distilbert"
    # 배포 환경(모델 미포함 레포)에서는 HF Hub에서 다운로드
    return SentimentClassifier(local if local.exists() else HUB_MODEL_ID)

@st.cache_data
def get_game_counts():
    """색인에서 게임별 리뷰 수 집계 (리뷰 많은 순), SQL GROUP BY로 직접 집계."""
    return _get_game_counts(min_count=MIN_GAME_REVIEWS)

@st.cache_data(show_spinner="리뷰 분류 중...")
def classify_reviews(reviews: tuple):
    """같은 리뷰 묶음(같은 게임)은 분류 결과를 캐시 — 재분석 시 즉시 반환."""
    from src.models.infer import split_by_sentiment
    results = get_classifier().predict(list(reviews))
    return split_by_sentiment(results)

def analyze_and_summarize(reviews):
    from src.llm.summarize import summarize
    pos, neg = classify_reviews(tuple(reviews))
    c1, c2 = st.columns(2)
    c1.metric("긍정", f"{len(pos)}건")
    c2.metric("부정", f"{len(neg)}건")
    with st.spinner("AI 요약 생성 중..."):
        st.markdown(summarize(pos, neg))
    return pos, neg

st.set_page_config(page_title="review-check", layout="wide")

# GRAYLUXE 스타일: 밝은 회색 베이스 · 얇은 타이포 · 소프트 블루 포인트
ACCENT = "#8FB9EA"
st.markdown(f"""
<style>
/* 섹션 킥커: 작은 대문자 파란 라벨 */
.kicker {{
    color: {ACCENT}; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 0.2rem;
}}
/* 페이지 타이틀: 얇은 큰 글씨, 포인트 단어만 파랑 */
.page-title {{
    font-size: 2.3rem; font-weight: 300; color: #E8EAEE;
    letter-spacing: -0.01em; line-height: 1.25; margin-bottom: 0.4rem;
}}
.page-title .accent {{ color: {ACCENT}; font-weight: 500; }}
.page-sub {{ color: #8A9099; font-size: 0.95rem; margin-bottom: 1.6rem; }}

/* 긍정/부정 지표: 흰 카드, 얇은 테두리, 여백 넉넉히 */
[data-testid="stMetric"] {{
    background: #282C33; border: 1px solid #3A3F47;
    border-radius: 14px; padding: 1.1rem 1.3rem;
}}
[data-testid="stMetricValue"] {{ font-weight: 300; }}

/* 버튼: 알약형 소프트 블루 */
.stButton > button, .stFormSubmitButton > button {{
    background: {ACCENT}; color: white; border: none;
    border-radius: 999px; padding: 0.45rem 1.6rem; font-weight: 500;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
    background: #7AAAE0; color: white;
}}

/* 입력·선택 상자: 흰 배경, 둥근 모서리 */
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {{
    background: #282C33; border-radius: 12px;
}}

/* 펼침 목록: 흰 카드 */
[data-testid="stExpander"] {{
    border: 1px solid #3A3F47; border-radius: 14px; background: #282C33;
}}

/* 사이드바: 흰색 + 얇은 경계 */
[data-testid="stSidebar"] {{
    border-right: 1px solid #3A3F47;
}}
.brand {{
    font-size: 1.15rem; font-weight: 600; letter-spacing: 0.18em;
    color: #E8EAEE; margin-bottom: 0.1rem;
}}
.brand-sub {{ color: #8A9099; font-size: 0.8rem; margin-bottom: 1.2rem; }}
</style>
""", unsafe_allow_html=True)

def page_header(kicker, title_html, sub):
    st.markdown(f'<div class="kicker">{kicker}</div>'
                f'<div class="page-title">{title_html}</div>'
                f'<div class="page-sub">{sub}</div>', unsafe_allow_html=True)

# 왼쪽 사이드바 대시보드 메뉴
with st.sidebar:
    st.markdown('<div class="brand">REVIEW CHECK</div>'
                '<div class="brand-sub">Steam 리뷰 감성분석 + AI 요약</div>',
                unsafe_allow_html=True)
    page = st.radio("메뉴", ["게임 분석", "리뷰 직접 입력", "리뷰 Q&A (RAG)",
                            "모델 정보", "RAG 구조"],
                    label_visibility="collapsed")
    st.divider()
    st.caption("DistilBERT 분류 · Qwen2.5 요약 · Supabase pgvector RAG")

if page == "게임 분석":
    page_header("GAME ANALYSIS",
                '게임의 <span class="accent">여론</span>을 한눈에',
                "게임을 고르면 리뷰 전체를 분류하고 총평과 장단점을 요약합니다.")
elif page == "리뷰 직접 입력":
    page_header("CUSTOM REVIEWS",
                '리뷰를 붙여넣어 <span class="accent">분석</span>하기',
                "어떤 리뷰든 한 줄에 하나씩 붙여넣으면 긍/부정을 판별하고 요약합니다.")
elif page == "리뷰 Q&A (RAG)":
    page_header("REVIEW Q&A",
                '리뷰에게 <span class="accent">질문</span>하세요',
                "질문과 의미가 가장 비슷한 리뷰를 찾아 근거와 함께 답합니다.")
elif page == "모델 정보":
    page_header("MODELS",
                '사용한 <span class="accent">모델</span> 정리',
                "이 데모를 구성하는 세 모델의 역할·파라미터·성능입니다.")
else:
    page_header("RAG PIPELINE",
                'RAG <span class="accent">구조</span> 한눈에 보기',
                "리뷰를 벡터로 저장해 두고, 질문이 오면 검색해 근거와 함께 답하는 구조입니다.")

if page == "게임 분석":
    games = get_game_counts()
    if not games:
        st.warning("색인에 게임 메타데이터가 없습니다. 06 노트북을 재실행하세요.")
    else:
        labels = [f"{name} — 리뷰 {n}건" for name, n in games]
        choice = st.selectbox("게임 선택", labels)
        if st.button("이 게임 분석", key="game_analyze"):
            game_name = games[labels.index(choice)][0]
            reviews = get_reviews_by_app(game_name)
            st.caption(f"'{game_name}' 리뷰 {len(reviews)}건을 분석합니다.")
            pos, neg = analyze_and_summarize(reviews)
            with st.expander("분석에 사용된 리뷰"):
                for r in reviews[:50]:
                    st.write(f"- {r[:300]}")

elif page == "리뷰 직접 입력":
    reviews_text = st.text_area("리뷰 (한 줄에 하나)", height=250)
    if st.button("분석", key="analyze"):
        reviews = [r.strip() for r in reviews_text.split("\n") if r.strip()]
        if not reviews:
            st.warning("리뷰를 입력하세요.")
        else:
            analyze_and_summarize(reviews)

elif page == "리뷰 Q&A (RAG)":
    games = get_game_counts()
    game_filter = st.selectbox("게임 선택", [name for name, _ in games])
    # form으로 묶으면 입력창에서 Enter만 눌러도 제출된다
    with st.form("qa_form"):
        question = st.text_input("질문", placeholder="예: is this game worth buying?")
        submitted = st.form_submit_button("질문")
    if submitted:
        if not question.strip():
            st.warning("질문을 입력하세요.")
        else:
            from src.rag.qa import answer
            with st.spinner("리뷰 검색 및 답변 생성 중..."):
                ans, contexts = answer(question, app_name=game_filter)
            st.markdown(ans)
            with st.expander("근거 리뷰"):
                for c in contexts:
                    st.write(f"- {c}")

elif page == "모델 정보":
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 감성 분류 — DistilBERT")
        st.markdown("""
| 항목 | 내용 |
|---|---|
| 모델 | `distilbert-base-uncased` 파인튜닝 |
| 파라미터 수 | 약 66M |
| 학습 데이터 | 균형 1만 건 (긍/부정 각 5천) |
| 학습 설정 | AdamW · lr 5e-5 · 유효 배치 16 · max_len 128 · 3에폭 (val F1 최적 선택) |
| 성능 (test) | **accuracy 0.856 · F1 0.855** |
| 배포 | HF Hub `rhantj/review-check-distilbert` |
""")
        st.caption("역할: 리뷰를 긍정/부정으로 분류 — 요약의 재료를 가른다.")
    with c2:
        st.markdown("#### 임베딩 — MiniLM")
        st.markdown("""
| 항목 | 내용 |
|---|---|
| 모델 | `all-MiniLM-L6-v2` (사전학습 그대로) |
| 파라미터 수 | 약 22M |
| 출력 | 문장 → 384차원 벡터 |
| 색인 | 리뷰 약 10만 건 → Supabase pgvector (`app_name`·`label` 메타데이터) |
| 검색 | 코사인 유사도 top-5 + 게임 필터 |
""")
        st.caption("역할: 질문과 의미가 비슷한 리뷰를 찾는다 (Q&A 검색).")
    with c3:
        st.markdown("#### 생성 — Qwen2.5")
        st.markdown("""
| 항목 | 내용 |
|---|---|
| 모델 | `Qwen/Qwen2.5-7B-Instruct` (HF Inference API) |
| 파라미터 수 | 약 7.6B |
| 폴백 | `Qwen/Qwen2.5-72B-Instruct` (1차 실패 시 자동 전환) |
| 호출 | LangChain 체인 (`prompt \\| llm \\| parser`) · max 512토큰 |
| 출력 | 총평 + 장점 3 + 단점 3 / 근거 기반 Q&A 답변 |
""")
        st.caption("역할: 받은 근거(분류 결과·검색 결과)를 한국어 문장으로 정리한다.")

    st.divider()
    st.markdown("""
**연결 구조** — 세 모델은 순서대로 협업합니다:
`리뷰 → DistilBERT(분류) → Qwen2.5(요약)` · `질문 → MiniLM(검색) → Qwen2.5(답변)`
""")

elif page == "RAG 구조":
    ACCENT_BG = "#253243"  # 어두운 파랑 틴트

    def flow(boxes):
        """박스 리스트를 화살표로 이은 가로 플로우 HTML."""
        items = []
        for label, desc, hl in boxes:
            bg = ACCENT_BG if hl else "#282C33"
            bd = ACCENT if hl else "#3A3F47"
            items.append(
                f'<div style="background:{bg};border:1px solid {bd};border-radius:12px;'
                f'padding:0.8rem 1rem;text-align:center;flex:1;min-width:120px;">'
                f'<div style="font-weight:600;font-size:0.9rem;color:#E8EAEE;">{label}</div>'
                f'<div style="font-size:0.75rem;color:#8A9099;margin-top:0.2rem;">{desc}</div></div>')
        arrow = f'<div style="color:{ACCENT};font-size:1.2rem;align-self:center;">→</div>'
        return ('<div style="display:flex;gap:0.6rem;align-items:stretch;margin:0.6rem 0 1.6rem;">'
                + arrow.join(items) + '</div>')

    st.markdown(f'<div class="kicker">STEP 1 · 색인 구축 — 한 번만 (노트북 06)</div>',
                unsafe_allow_html=True)
    st.markdown(flow([
        ("리뷰 약 10만 건", "긍정·부정 균형 샘플", False),
        ("임베딩 모델", "MiniLM · 문장→384차원 벡터", True),
        ("Supabase 저장", "pgvector + 원문 + 게임명·라벨", True),
    ]), unsafe_allow_html=True)

    st.markdown(f'<div class="kicker">STEP 2 · 질의응답 — 질문할 때마다 (리뷰 Q&A 메뉴)</div>',
                unsafe_allow_html=True)
    st.markdown(flow([
        ("질문 입력", '"is this game worth buying?"', False),
        ("질문 임베딩", "같은 모델로 벡터 변환", True),
        ("벡터 검색", "게임 필터 + 의미가 비슷한 리뷰 5개", True),
        ("LLM 생성", "Qwen2.5 · 근거로만 답하라", True),
        ("답변 + 근거", "근거 리뷰 원문 함께 표시", False),
    ]), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 왜 이렇게 하나")
        st.markdown("""
- LLM에게 그냥 물으면 **지어낸 답(환각)** 이 나올 수 있음
- 실제 리뷰를 검색해 근거로 건네면 **데이터에 있는 내용만** 답하게 됨
- "근거가 없으면 모른다고 답하라"는 지시 + 근거 원문 노출로 검증 가능
""")
    with c2:
        st.markdown("##### 검색이 키워드가 아닌 '의미'로 되는 이유")
        st.markdown("""
- 임베딩 모델이 의미가 비슷한 문장을 **가까운 벡터**로 만들어 둠
- 질문도 같은 공간의 벡터로 바꿔 **가장 가까운 리뷰**를 찾음
- 그래서 "buggy?"로 물어도 "crashes constantly" 리뷰가 검색됨
""")
