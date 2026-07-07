import psycopg
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector

from src.config import DATABASE_URL, EMBED_MODEL_ID

COLLECTION_NAME = "reviews"
_embeddings = None


def get_embeddings():
    """임베딩 모델 (최초 1회만 로드)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    return _embeddings


def _pg_connection_string(url):
    """langchain-postgres는 psycopg3 드라이버 스킴이 필요하다 (postgresql+psycopg://)."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_vectorstore(connection=None):
    """조회용 PGVector 벡터스토어 (기본: 앱 런타임 pooler 연결)."""
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=_pg_connection_string(connection or DATABASE_URL),
        use_jsonb=True,
    )


def reset_index(connection):
    """기존 색인을 비우고 빈 컬렉션을 새로 만든다 (재구축 시작 시 1회만 호출)."""
    PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=_pg_connection_string(connection),
        use_jsonb=True,
        pre_delete_collection=True,
    )


def add_batch(texts, metadatas, connection):
    """색인에 배치를 추가한다 (재시작 가능하도록 작은 단위로 반복 호출)."""
    vs = PGVector(
        embeddings=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection=_pg_connection_string(connection),
        use_jsonb=True,
    )
    vs.add_texts(texts, metadatas=metadatas)


def get_game_counts(min_count=20):
    """게임별 리뷰 수 집계 (많은 순), min_count 미만은 제외."""
    query = """
        SELECT e.cmetadata->>'app_name' AS app_name, COUNT(*) AS n
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
          AND e.cmetadata->>'app_name' IS NOT NULL
          AND e.cmetadata->>'app_name' != '(unknown)'
        GROUP BY 1
        HAVING COUNT(*) >= %s
        ORDER BY n DESC
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (COLLECTION_NAME, min_count))
            return cur.fetchall()


def get_reviews_by_app(app_name):
    """주어진 게임의 리뷰 원문 전체 조회."""
    query = """
        SELECT e.document
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s AND e.cmetadata->>'app_name' = %s
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (COLLECTION_NAME, app_name))
            return [row[0] for row in cur.fetchall()]
