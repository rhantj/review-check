# Supabase pgvector 마이그레이션 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RAG 벡터 저장소를 로컬 Chroma에서 Supabase(Postgres + pgvector)로 교체하고, 색인 규모를 80만 건(긍정 40만 + 부정 40만)으로 확장한다.

**Architecture:** `langchain-postgres`의 `PGVector`를 통해 Supabase Postgres에 직접 연결(SDK/REST 아님)한다. 색인 구축(임베딩+대량 insert)은 `notebooks/06_build_index.ipynb`에서 1회 배치로 수행하고, `src/rag/index.py`가 조회(벡터 검색 + 게임별 집계 SQL)를 전담한다. `src/rag/qa.py`, `app.py`는 이 인터페이스만 소비한다.

**Tech Stack:** `langchain-postgres`, `psycopg[binary]`, Supabase Postgres(pgvector 확장), `langchain-huggingface`(임베딩), Streamlit

## Global Constraints

- 색인 샘플 규모: 긍정 40만 + 부정 40만 = 80만 건 (spec 확정치)
- Chroma는 완전히 제거한다 — 환경변수 토글 없이 pgvector 단일화
- Supabase 접속은 직접 Postgres 연결(`langchain-postgres` PGVector)만 사용, REST/anon API 키는 쓰지 않는다
- 임베딩 계산은 로컬 Mac에서 배치로 1회 실행 (`all-MiniLM-L6-v2`)
- `tests/` 디렉터리에 파일을 만들지 않는다 — 검증은 수동 스크립트/명령으로 한다 (CLAUDE.md 프로젝트 규칙)
- 노트북 실행은 사용자가 직접 한다 — Claude는 노트북 코드 작성까지만 한다
- 이번 작업 전체는 사용자가 명시적으로 지시할 때까지 git에 커밋/푸시하지 않는다
- `chroma_store/` 삭제 및 `.gitignore` 정리는 최종 검증 통과 + 사용자의 별도 실행 지시 이후에만 수행한다 (파일 삭제는 확인 후 실행 대상)
- 두 개의 접속 문자열이 필요하다: `DATABASE_URL`(6543 pooler, 앱 런타임용), `DATABASE_URL_DIRECT`(5432 직접 연결, 노트북 대량 적재/DDL용) — `.env`에 사용자가 직접 채워 넣는다

---

## 파일 구조 개요

| 파일 | 역할 |
|---|---|
| `requirements.txt` | `chromadb`/`langchain-chroma` 제거, `langchain-postgres`/`psycopg[binary]` 추가 |
| `src/config.py` | `VECTOR_DIR` 제거, `DATABASE_URL`/`DATABASE_URL_DIRECT` 추가 |
| `src/rag/index.py` | Chroma → PGVector 전면 교체. 조회용 `get_vectorstore()`, 대량 적재용 `reset_index()`/`add_batch()`, 집계/조회 SQL `get_game_counts()`/`get_reviews_by_app()` |
| `src/rag/qa.py` | `get_vectorstore()` 호출부만 인자 제거 (그 외 변경 없음) |
| `app.py` | `get_collection()` 삭제, 게임 집계/리뷰 조회를 `src.rag.index`의 새 함수로 교체, UI 텍스트(건수/저장소 이름) 갱신 |
| `notebooks/06_build_index.ipynb` | 전면 교체 — 80만 건 샘플링 → 정제 → 배치 임베딩+적재(체크포인트 재개 지원) → HNSW 인덱스 생성 |

---

### Task 1: 의존성 갱신

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: 없음
- Produces: `langchain_postgres.PGVector`, `psycopg` 임포트가 이후 태스크에서 사용 가능해짐

- [ ] **Step 1: requirements.txt에서 Chroma 관련 줄 제거, Postgres 관련 줄 추가**

`chromadb>=1.5`, `langchain-chroma>=1.1` 두 줄을 삭제하고, `langchain-ollama>=1.1` 다음 줄에 아래 두 줄을 추가한다.

```
langchain-postgres>=0.0.14
psycopg[binary]>=3.2
```

**Modify** `requirements.txt` 전체 내용이 다음과 같아지도록 한다:

```
torch>=2.12
torchvision>=0.27
transformers>=5.12
datasets>=3.0
accelerate>=1.0
scikit-learn>=1.7
sentence-transformers>=5.6
streamlit>=1.40
huggingface-hub>=0.27
python-dotenv>=1.2
pandas>=2.2
matplotlib>=3.10
pytest>=8.0
langchain-core>=1.4
langchain-huggingface>=1.2
langchain-postgres>=0.0.14
langchain-ollama>=1.1
psycopg[binary]>=3.2
```

- [ ] **Step 2: 의존성 설치**

```bash
pip install -r requirements.txt
```
Expected: `langchain-postgres`, `psycopg` 설치 완료, 에러 없음.

- [ ] **Step 3: 커밋 (로컬 커밋만, push 금지)**

이번 마이그레이션 전체는 사용자가 별도로 git 반영을 지시할 때까지 커밋하지 않는다. 이 스텝은 건너뛴다.

---

### Task 2: `src/config.py` 갱신

**Files:**
- Modify: `src/config.py`

**Interfaces:**
- Consumes: 없음
- Produces: `DATABASE_URL`, `DATABASE_URL_DIRECT` (str | None) — Task 3, 5, 7에서 사용

- [ ] **Step 1: `VECTOR_DIR` 제거, DB 접속 문자열 상수 추가**

`src/config.py` 전체를 아래로 교체한다:

```python
import os
from pathlib import Path

RANDOM_SEED = 42
MAX_LEN = 128  # 추론 토큰 상한 — 학습(TRAIN_MAX_LEN=128)과 일치, 초과분은 낭비

HUB_MODEL_ID = "rhantj/review-check-distilbert"  # 로컬 models/distilbert 부재 시 폴백
LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LLM_FALLBACK_ID = "Qwen/Qwen2.5-72B-Instruct"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K = 5

DATABASE_URL = os.environ.get("DATABASE_URL")  # 앱 런타임용 (Supabase transaction pooler, 6543)
DATABASE_URL_DIRECT = os.environ.get("DATABASE_URL_DIRECT")  # 노트북 대량 적재·DDL용 (직접 연결, 5432)

LABELS = {0: "부정", 1: "긍정"}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"
```

- [ ] **Step 2: import 확인**

```bash
python -c "from src.config import DATABASE_URL, DATABASE_URL_DIRECT; print('ok')"
```
Expected: `ok` 출력 (환경변수가 아직 없어도 `None`으로 로드되므로 에러 없음).

- [ ] **Step 3: `VECTOR_DIR`을 참조하는 다른 파일이 없는지 확인**

```bash
grep -rn "VECTOR_DIR" --include="*.py" .
```
Expected: 결과 없음(전부 Task 4, 5에서 제거 예정이므로, 이 스텝은 Task 4·5 완료 후 최종 확인용으로 다시 실행해도 됨).

---

### Task 3: `src/rag/index.py` 전면 교체

**Files:**
- Modify: `src/rag/index.py`

**Interfaces:**
- Consumes: `src.config.DATABASE_URL`, `src.config.EMBED_MODEL_ID`
- Produces:
  - `get_embeddings() -> HuggingFaceEmbeddings`
  - `get_vectorstore(connection: str | None = None) -> PGVector` (기본: `DATABASE_URL`)
  - `reset_index(connection: str) -> None` (컬렉션 초기화, 노트북 전용)
  - `add_batch(texts: list[str], metadatas: list[dict] | None, connection: str) -> None` (노트북 전용)
  - `get_game_counts(min_count: int = 20) -> list[tuple[str, int]]`
  - `get_reviews_by_app(app_name: str) -> list[str]`

- [ ] **Step 1: `src/rag/index.py` 전체를 아래로 교체**

```python
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
```

- [ ] **Step 2: import 및 문법 확인 (DB 연결 없이 가능)**

```bash
python -c "from src.rag import index; print([f for f in dir(index) if not f.startswith('_')])"
```
Expected: `['COLLECTION_NAME', 'add_batch', 'get_embeddings', 'get_game_counts', 'get_reviews_by_app', 'get_vectorstore', 'reset_index', ...]` 형태 출력, 에러 없음.

> DB 연결이 필요한 함수(`get_vectorstore`, `get_game_counts`, `get_reviews_by_app`)는 Task 6에서 `.env`에 실제 접속 문자열을 넣은 뒤 Task 8에서 실행 검증한다.

---

### Task 4: `src/rag/qa.py` 갱신

**Files:**
- Modify: `src/rag/qa.py`

**Interfaces:**
- Consumes: `src.rag.index.get_vectorstore()` (Task 3에서 시그니처가 `connection=None` 기본 인자로 바뀜)
- Produces: `answer(question: str, app_name: str | None = None) -> tuple[str, list[str]]` (시그니처 변경 없음)

- [ ] **Step 1: `VECTOR_DIR` import 제거, `get_vectorstore()` 호출을 무인자로 변경**

`src/rag/qa.py`의 상단 import를:

```python
from src.config import VECTOR_DIR, RAG_TOP_K
```

다음으로 교체:

```python
from src.config import RAG_TOP_K
```

그리고 다음 줄을:

```python
    retriever = get_vectorstore(VECTOR_DIR).as_retriever(search_kwargs=search_kwargs)
```

다음으로 교체:

```python
    retriever = get_vectorstore().as_retriever(search_kwargs=search_kwargs)
```

- [ ] **Step 2: import 확인**

```bash
python -c "from src.rag import qa; print('ok')"
```
Expected: `ok` 출력, 에러 없음.

---

### Task 5: `app.py` 갱신

**Files:**
- Modify: `app.py:1-31` (import, `get_collection`/`get_game_counts`), `app.py:123`, `app.py:154-158`, `app.py:215`, `app.py:258-262`

**Interfaces:**
- Consumes: `src.rag.index.get_game_counts(min_count)`, `src.rag.index.get_reviews_by_app(app_name)`
- Produces: 없음 (최종 소비자)

- [ ] **Step 1: import 및 캐시 함수 교체**

`app.py` 3번째 줄:

```python
from src.config import MODEL_DIR, VECTOR_DIR, HUB_MODEL_ID
```

다음으로 교체:

```python
from src.config import MODEL_DIR, HUB_MODEL_ID
from src.rag.index import get_game_counts as _get_game_counts, get_reviews_by_app
```

17~31번째 줄(`get_collection()`, `get_game_counts()` 두 함수) 전체를:

```python
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
```

다음으로 교체:

```python
@st.cache_data
def get_game_counts():
    """색인에서 게임별 리뷰 수 집계 (리뷰 많은 순), SQL GROUP BY로 직접 집계."""
    return _get_game_counts(min_count=MIN_GAME_REVIEWS)
```

- [ ] **Step 2: 게임 분석 탭에서 리뷰 조회 방식 교체**

`app.py`의 "게임 분석" 탭 블록에서:

```python
        if st.button("이 게임 분석", key="game_analyze"):
            game_name = games[labels.index(choice)][0]
            got = get_collection().get(where={"app_name": game_name},
                                       include=["documents"])
            reviews = got["documents"]
            st.caption(f"'{game_name}' 리뷰 {len(reviews)}건을 분석합니다.")
```

다음으로 교체:

```python
        if st.button("이 게임 분석", key="game_analyze"):
            game_name = games[labels.index(choice)][0]
            reviews = get_reviews_by_app(game_name)
            st.caption(f"'{game_name}' 리뷰 {len(reviews)}건을 분석합니다.")
```

- [ ] **Step 3: 사이드바 캡션 갱신**

```python
    st.caption("DistilBERT 분류 · Qwen2.5 요약 · Chroma RAG")
```

다음으로 교체:

```python
    st.caption("DistilBERT 분류 · Qwen2.5 요약 · Supabase pgvector RAG")
```

- [ ] **Step 4: "모델 정보" 탭 색인 설명 갱신**

```python
| 색인 | 리뷰 10,810건 → Chroma (`app_name`·`label` 메타데이터) |
```

다음으로 교체:

```python
| 색인 | 리뷰 80만 건 → Supabase pgvector (`app_name`·`label` 메타데이터) |
```

- [ ] **Step 5: "RAG 구조" 탭 플로우 박스 갱신**

```python
    st.markdown(flow([
        ("리뷰 10,810건", "train 전체 + 인기 20개 게임", False),
        ("임베딩 모델", "MiniLM · 문장→384차원 벡터", True),
        ("Chroma 저장", "벡터 + 원문 + 게임명·라벨", True),
    ]), unsafe_allow_html=True)
```

다음으로 교체:

```python
    st.markdown(flow([
        ("리뷰 80만 건", "긍정 40만 + 부정 40만 균형 샘플", False),
        ("임베딩 모델", "MiniLM · 문장→384차원 벡터", True),
        ("Supabase 저장", "pgvector + 원문 + 게임명·라벨", True),
    ]), unsafe_allow_html=True)
```

- [ ] **Step 6: import 확인**

```bash
python -c "import ast; ast.parse(open('app.py').read()); print('ok')"
```
Expected: `ok` 출력, 문법 에러 없음.

- [ ] **Step 7: `VECTOR_DIR`/Chroma 잔재 확인**

```bash
grep -n "VECTOR_DIR\|chromadb\|Chroma" app.py src/rag/*.py src/config.py
```
Expected: 결과 없음.

---

### Task 6: `.env`에 Supabase 접속 정보 입력 (사용자 액션)

**Files:**
- Modify: `.env` (사용자가 직접 편집, gitignore 대상이라 Claude가 값을 대신 채우지 않음)

- [ ] **Step 1: 사용자에게 알림**

이 태스크 시작 시 사용자에게 다음을 전달하고 입력을 기다린다:

> Supabase 대시보드 → Project Settings → Database → Connection string에서 아래 두 값을 확인해 `.env`에 추가해 주세요.
> ```
> DATABASE_URL=postgresql://postgres.[project-ref]:[비밀번호]@aws-0-[region].pooler.supabase.com:6543/postgres
> DATABASE_URL_DIRECT=postgresql://postgres:[비밀번호]@db.[project-ref].supabase.co:5432/postgres
> ```

- [ ] **Step 2: 값이 채워졌는지 확인 (내용은 출력하지 않고 존재 여부만 확인)**

```bash
python -c "
from dotenv import load_dotenv
load_dotenv()
import os
print('DATABASE_URL set:', bool(os.environ.get('DATABASE_URL')))
print('DATABASE_URL_DIRECT set:', bool(os.environ.get('DATABASE_URL_DIRECT')))
"
```
Expected: 둘 다 `True`.

- [ ] **Step 3: 실제 접속 확인**

```bash
python -c "
from dotenv import load_dotenv
load_dotenv()
import psycopg, os
with psycopg.connect(os.environ['DATABASE_URL_DIRECT']) as conn:
    with conn.cursor() as cur:
        cur.execute('SELECT extname FROM pg_extension;')
        print([r[0] for r in cur.fetchall()])
"
```
Expected: 리스트에 `vector`가 포함됨. 없다면 Supabase SQL Editor에서 `CREATE EXTENSION IF NOT EXISTS vector;` 실행 후 재시도.

---

### Task 7: `notebooks/06_build_index.ipynb` 전면 교체

**Files:**
- Modify: `notebooks/06_build_index.ipynb`
- Create (체크포인트 파일, 노트북 실행 중 생성): `output/06_pgvector_checkpoint.txt`

**Interfaces:**
- Consumes: `src.data.pipeline.clean_reviews`, `src.rag.index.reset_index/add_batch/get_game_counts`, `src.config.DATABASE_URL_DIRECT`
- Produces: Supabase의 `reviews` 컬렉션에 80만 건 색인 완료 + HNSW 인덱스

이 태스크는 Claude가 노트북 셀 내용을 작성하는 것까지이며, **실행은 사용자가 직접** 한다 (CLAUDE.md 정책).

- [ ] **Step 1: 노트북 셀 구성 — 아래 순서로 코드 셀을 작성한다**

셀 1 (환경 로드):
```python
import os, time
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
from src.config import DATABASE_URL_DIRECT, RANDOM_SEED
from src.data.pipeline import clean_reviews
from src.rag.index import reset_index, add_batch, get_game_counts

N_PER_LABEL = 400_000
BATCH_SIZE = 5_000
CHECKPOINT_PATH = "../output/06_pgvector_checkpoint.txt"
```

셀 2 (원본 로드 + 균형 샘플링):
```python
ds = load_dataset("ksang/steamreviews", split="train")
df = ds.to_pandas()

pos = df[df["review_score"] == 1].sample(n=N_PER_LABEL, random_state=RANDOM_SEED)
neg = df[df["review_score"] == -1].sample(n=N_PER_LABEL, random_state=RANDOM_SEED)
sample = pd.concat([pos, neg]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
print(len(sample))  # 800000 이어야 함
```

셀 3 (정제):
```python
cleaned = clean_reviews(sample, extra_cols=["app_name"])
print(len(cleaned))
```

셀 4 (체크포인트 기반 배치 적재 — 중단 시 재실행하면 이어서 진행):
```python
texts = cleaned["text"].tolist()
metadatas = [{"app_name": row.app_name, "label": int(row.label)}
             for row in cleaned.itertuples()]

start_batch = 0
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH) as f:
        start_batch = int(f.read().strip())
else:
    reset_index(DATABASE_URL_DIRECT)  # 최초 1회만 컬렉션 초기화

for s in range(start_batch, len(texts), BATCH_SIZE):
    e = min(s + BATCH_SIZE, len(texts))
    t0 = time.time()
    add_batch(texts[s:e], metadatas[s:e], DATABASE_URL_DIRECT)
    with open(CHECKPOINT_PATH, "w") as f:
        f.write(str(e))
    print(f"{e}/{len(texts)} ({time.time()-t0:.1f}s)")
```

셀 5 (HNSW 인덱스 생성 — 적재 완료 후 1회 실행):
```python
import psycopg

with psycopg.connect(DATABASE_URL_DIRECT) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS langchain_pg_embedding_hnsw_idx
            ON langchain_pg_embedding
            USING hnsw (embedding vector_cosine_ops);
        """)
        cur.execute("ANALYZE langchain_pg_embedding;")
print("index created")
```

셀 6 (검증 — 게임별 집계):
```python
counts = get_game_counts(min_count=20)
print(len(counts), counts[:10])
```

- [ ] **Step 2: 사용자에게 실행 요청**

사용자에게 다음을 전달한다:

> `notebooks/06_build_index.ipynb`를 새 내용으로 작성했습니다. 80만 건 임베딩은 로컬에서 수 시간 걸릴 수 있고, 중단돼도 재실행하면 체크포인트(`output/06_pgvector_checkpoint.txt`)부터 이어집니다. 직접 실행해 주시고, 끝나면 알려주세요.

---

### Task 8: 로컬 앱 End-to-End 검증

**Files:**
- 없음 (실행/관찰만)

**Interfaces:**
- Consumes: Task 7 완료 후의 Supabase 색인, Task 3~5의 코드

- [ ] **Step 1: Streamlit 앱 로컬 실행**

```bash
streamlit run app.py
```

- [ ] **Step 2: "게임 분석" 탭 확인**

브라우저에서 게임 목록이 뜨는지, 게임 하나를 선택해 분석 버튼을 누르면 긍/부정 건수와 AI 요약이 나오는지 확인.
Expected: 오류 없이 결과 표시.

- [ ] **Step 3: "리뷰 Q&A (RAG)" 탭 확인 — 필터 검색 정합성 확인**

게임을 하나 선택하고 질문을 입력해 답변과 "근거 리뷰"가 실제로 그 게임의 리뷰인지 확인.
Expected: 근거 리뷰의 게임이 선택한 게임과 일치. (불일치 시 `src/rag/index.py`의 `get_vectorstore().as_retriever(search_kwargs={"filter": {...}})` 필터 문법을 langchain-postgres 최신 문서 기준으로 재확인 — 예: `{"app_name": {"$eq": app_name}}` 형태로 조정 필요할 수 있음)

- [ ] **Step 4: "모델 정보"/"RAG 구조" 탭 텍스트 확인**

건수(80만 건)와 저장소 이름(Supabase pgvector)이 올바르게 표시되는지 확인.

- [ ] **Step 5: 문제 없으면 사용자에게 보고**

앱이 정상 동작하면 사용자에게 "검증 완료, Task 9(Chroma 제거) 진행해도 될지" 확인을 요청한다.

---

### Task 9: Chroma 제거 (게이트 — 사용자 명시적 지시 시에만 실행)

**Files:**
- Delete: `chroma_store/`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Task 8 검증 통과

- [ ] **Step 1: 사용자 확인 대기**

Task 8 검증이 끝나고 사용자가 "Chroma 지워도 돼" 등으로 명시적으로 지시하기 전까지는 이 태스크를 실행하지 않는다.

- [ ] **Step 2: (지시 받은 후) `chroma_store/` 삭제**

```bash
rm -rf chroma_store/
```

- [ ] **Step 3: `.gitignore`에서 Chroma 예외 규칙 제거**

`.gitignore`에서 다음 두 줄을 삭제한다:

```
# 배포(Streamlit Cloud 등)에서 RAG 색인이 필요해 chroma_store는 레포에 포함
!chroma_store/**
```

- [ ] **Step 4: git 반영 여부는 별도 지시 대기**

이번 마이그레이션 전체와 마찬가지로, git add/commit은 사용자가 명시적으로 지시할 때에만 수행한다.

---

## Self-Review 체크리스트 (계획 작성자용, 참고)

- 스펙의 모든 결정 사항(80만 건, Chroma 완전 제거, 직접 Postgres 연결, 로컬 배치 임베딩, 롤아웃 순서, 환경변수 두 개, git 보류)이 Task 1~9에 반영됨
- `get_vectorstore()` 시그니처 변경(`persist_dir` 위치인자 → `connection` 키워드 기본 `None`)이 `qa.py`(Task 4)와 일치
- `build_index()` 단일 함수 대신 `reset_index()`/`add_batch()`로 분리한 이유(체크포인트 재개 지원)를 Task 3/7에 명시
- PGVector의 metadata 필터 문법은 실제 라이브러리 버전에 따라 달라질 수 있어 Task 8에서 실측 검증 스텝을 명시적으로 둠(placeholder 아님 — 구체적 대안 문법까지 제시)
