import streamlit as st
from dotenv import load_dotenv
from src.config import MODEL_DIR
from src.models.infer import SentimentClassifier, split_by_sentiment
from src.llm.summarize import summarize

load_dotenv()

@st.cache_resource
def get_classifier():
    return SentimentClassifier(MODEL_DIR / "distilbert")

st.set_page_config(page_title="review-check", layout="wide")
st.title("review-check — Steam 리뷰 감성분석 + AI 요약")

tab1, tab2 = st.tabs(["감성분석 + 요약", "리뷰 Q&A (RAG)"])

with tab1:
    reviews_text = st.text_area("리뷰 (한 줄에 하나)", height=250)
    if st.button("분석", key="analyze"):
        reviews = [r.strip() for r in reviews_text.split("\n") if r.strip()]
        if not reviews:
            st.warning("리뷰를 입력하세요.")
        else:
            clf = get_classifier()
            results = clf.predict(reviews)
            pos, neg = split_by_sentiment(results)
            c1, c2 = st.columns(2)
            c1.metric("긍정", f"{len(pos)}건")
            c2.metric("부정", f"{len(neg)}건")
            with st.spinner("AI 요약 생성 중..."):
                st.markdown(summarize(pos, neg))

with tab2:
    question = st.text_input("질문", placeholder="예: 이 게임 멀미 있어?")
    if st.button("질문", key="ask"):
        if not question.strip():
            st.warning("질문을 입력하세요.")
        else:
            from src.rag.qa import answer
            with st.spinner("리뷰 검색 및 답변 생성 중..."):
                ans, contexts = answer(question)
            st.markdown(ans)
            with st.expander("근거 리뷰"):
                for c in contexts:
                    st.write(f"- {c}")
