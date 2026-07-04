import streamlit as st
from dotenv import load_dotenv
from src.config import MODEL_DIR, VECTOR_DIR, HUB_MODEL_ID
from src.models.infer import SentimentClassifier, split_by_sentiment
from src.llm.summarize import summarize

load_dotenv()

MIN_GAME_REVIEWS = 20  # 게임 분석 탭에 노출할 최소 리뷰 수

@st.cache_resource
def get_classifier():
    local = MODEL_DIR / "distilbert"
    # 배포 환경(모델 미포함 레포)에서는 HF Hub에서 다운로드
    return SentimentClassifier(local if local.exists() else HUB_MODEL_ID)

@st.cache_resource
def get_collection():
    from src.rag.index import get_collection
    return get_collection(VECTOR_DIR)

@st.cache_data
def get_game_counts():
    """색인 메타데이터에서 게임별 리뷰 수 집계 (리뷰 많은 순)."""
    from collections import Counter
    got = get_collection().get(include=["metadatas"])
    counts = Counter(m["app_name"] for m in got["metadatas"]
                     if m and m.get("app_name") and m["app_name"] != "(unknown)")
    return [(name, n) for name, n in counts.most_common() if n >= MIN_GAME_REVIEWS]

def analyze_and_summarize(reviews):
    results = get_classifier().predict(reviews)
    pos, neg = split_by_sentiment(results)
    c1, c2 = st.columns(2)
    c1.metric("긍정", f"{len(pos)}건")
    c2.metric("부정", f"{len(neg)}건")
    with st.spinner("AI 요약 생성 중..."):
        st.markdown(summarize(pos, neg))
    return pos, neg

st.set_page_config(page_title="review-check", layout="wide")
st.title("review-check — Steam 리뷰 감성분석 + AI 요약")

tab1, tab2, tab3 = st.tabs(["게임 분석", "리뷰 직접 입력", "리뷰 Q&A (RAG)"])

with tab1:
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

with tab2:
    reviews_text = st.text_area("리뷰 (한 줄에 하나)", height=250)
    if st.button("분석", key="analyze"):
        reviews = [r.strip() for r in reviews_text.split("\n") if r.strip()]
        if not reviews:
            st.warning("리뷰를 입력하세요.")
        else:
            analyze_and_summarize(reviews)

with tab3:
    games = get_game_counts()
    game_filter = st.selectbox("게임 선택", [name for name, _ in games])
    question = st.text_input("질문", placeholder="예: is this game worth buying?")
    if st.button("질문", key="ask"):
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
