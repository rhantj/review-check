import streamlit as st
from dotenv import load_dotenv
from src.config import MODEL_DIR, VECTOR_DIR, HUB_MODEL_ID

load_dotenv()

MIN_GAME_REVIEWS = 20  # 게임 분석 탭에 노출할 최소 리뷰 수

@st.cache_resource
def get_classifier():
    # torch/transformers는 무거우므로 실제 분석 시점에만 import (기동 메모리 절약)
    from src.models.infer import SentimentClassifier
    local = MODEL_DIR / "distilbert"
    # 배포 환경(모델 미포함 레포)에서는 HF Hub에서 다운로드
    return SentimentClassifier(local if local.exists() else HUB_MODEL_ID)

@st.cache_resource
def get_collection():
    """메타데이터·문서 조회 전용 — 임베딩 모델 없이 Chroma만 연다.
    (임베딩 모델은 Q&A에서 실제 검색할 때만 로드해 배포 메모리를 아낀다)"""
    import chromadb
    return chromadb.PersistentClient(path=str(VECTOR_DIR)).get_collection("reviews")

@st.cache_data
def get_game_counts():
    """색인 메타데이터에서 게임별 리뷰 수 집계 (리뷰 많은 순)."""
    from collections import Counter
    got = get_collection().get(include=["metadatas"])
    counts = Counter(m["app_name"] for m in got["metadatas"]
                     if m and m.get("app_name") and m["app_name"] != "(unknown)")
    return [(name, n) for name, n in counts.most_common() if n >= MIN_GAME_REVIEWS]

def analyze_and_summarize(reviews):
    from src.models.infer import split_by_sentiment
    from src.llm.summarize import summarize
    results = get_classifier().predict(reviews)
    pos, neg = split_by_sentiment(results)
    c1, c2 = st.columns(2)
    c1.metric("긍정", f"{len(pos)}건")
    c2.metric("부정", f"{len(neg)}건")
    with st.spinner("AI 요약 생성 중..."):
        st.markdown(summarize(pos, neg))
    return pos, neg

st.set_page_config(page_title="review-check", layout="wide")

# 회색 베이스 + 옅은 파란색 포인트 (기본 팔레트는 .streamlit/config.toml)
st.markdown("""
<style>
/* 페이지 제목 아래 포인트 라인 */
h1 { border-bottom: 3px solid #6C9BD1; padding-bottom: 0.35rem; }

/* 긍정/부정 지표를 카드로 */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D8DCE2;
    border-left: 4px solid #6C9BD1;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
}

/* 버튼: 파란 포인트, 호버 시 살짝 진하게 */
.stButton > button, .stFormSubmitButton > button {
    background: #6C9BD1; color: white; border: none; border-radius: 8px;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    background: #5A89C4; color: white;
}

/* 사이드바 구분감 */
[data-testid="stSidebar"] { border-right: 1px solid #D8DCE2; }

/* 펼침 목록(근거 리뷰 등) 테두리 정돈 */
[data-testid="stExpander"] {
    border: 1px solid #D8DCE2; border-radius: 10px; background: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# 왼쪽 사이드바 대시보드 메뉴
with st.sidebar:
    st.title("Review Check")
    st.caption("Steam 리뷰 감성분석 + AI 요약")
    page = st.radio("메뉴", ["게임 분석", "리뷰 직접 입력", "리뷰 Q&A (RAG)"],
                    label_visibility="collapsed")
    st.divider()
    st.caption("DistilBERT 분류 · Qwen2.5 요약 · Chroma RAG")

st.title(page)

if page == "게임 분석":
    games = get_game_counts()
    if not games:
        st.warning("색인에 게임 메타데이터가 없습니다. 06 노트북을 재실행하세요.")
    else:
        labels = [f"{name} — 리뷰 {n}건" for name, n in games]
        choice = st.selectbox("게임 선택", labels)
        if st.button("이 게임 분석", key="game_analyze"):
            game_name = games[labels.index(choice)][0]
            got = get_collection().get(where={"app_name": game_name},
                                       include=["documents"])
            reviews = got["documents"]
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
