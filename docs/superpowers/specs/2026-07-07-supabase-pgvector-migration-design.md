# Supabase pgvector 마이그레이션 설계

작성일: 2026-07-07

## 배경 / 목적

현재 RAG 색인은 로컬 파일 기반 Chroma(`chroma_store/`, 102MB, 10,810건: train 5,686 + 인기 게임 20개 부스팅 5,146건)를 사용하며, git 저장소에 그대로 포함시켜 Streamlit Cloud 배포 시 같이 실어 나른다.

전체 데이터셋(HF `ksang/steamreviews`, 약 641만 건: 긍정 526만/부정 116만)을 활용한 검색 커버리지를 원하는데, git-내장 파일 스토어로는 규모/용량 한계가 있다. 이번 마이그레이션은 벡터 저장소를 **Supabase(Postgres + pgvector)**로 옮겨 이 제약을 해소한다.

## 결정된 범위

- **샘플 규모**: 완전 반반 균형 최대치(부정 116만 전체)가 아니라, 사용자 결정에 따라 **80만 건(긍정 40만 + 부정 40만)**으로 축소. 예상 저장 용량 약 9~10GB 수준(전체 641만 대비 절충).
- **Chroma는 완전 제거**, 환경변수 토글 없이 pgvector로 일원화한다(코드 단순화 우선, YAGNI).
- **접속 방식**: Supabase 클라이언트 SDK(`SupabaseVectorStore` + RPC) 대신 **직접 Postgres 연결**(`langchain-postgres`의 `PGVector`)을 사용한다. Streamlit Cloud에서도 외부 TCP 연결이 가능하므로 문제 없음.
- **임베딩 계산 위치**: 로컬 Mac에서 배치로 1회 실행(추가 비용 없음, 시간만 소요). HF Inference API 등 외부 임베딩 호출은 사용하지 않는다.
- 기존 감성분류 학습용 데이터(`data/{train,val,test}.csv`, 8,126건)는 이번 변경과 무관하며 그대로 둔다. 이번 변경은 RAG 색인용 샘플에만 적용된다.
- 기존에 있던 "인기 게임 20개 리뷰 추가 부스팅" 로직은 제거한다. 표본이 80만 건으로 커지면서 인기 게임 리뷰도 충분히 포함되므로 별도 부스팅이 불필요하다.
- **git 반영은 보류** — 이번 작업 전체(코드/노트북 변경, `chroma_store/` 삭제 등)는 사용자가 명시적으로 지시할 때까지 커밋/푸시하지 않는다.

## 아키텍처

```
[HF ksang/steamreviews 전체]
        │  균형 샘플링 80만 건 (긍정 40만 + 부정 40만)
        ▼
[src.data.pipeline.clean_reviews(extra_cols=["app_name"])]
        │  정제
        ▼
[all-MiniLM-L6-v2 로컬 배치 임베딩]  (notebooks/06_build_index.ipynb)
        │  5,000건 단위 batch insert
        ▼
[Supabase Postgres + pgvector]
   - langchain-postgres PGVector 기본 스키마
     (langchain_pg_collection, langchain_pg_embedding)
   - embedding: vector(384), metadata: JSONB(app_name, label)
   - CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)
        │
        ▼
[src/rag/qa.py]  retriever.as_retriever(search_kwargs={"filter": {"app_name": ...}})
[app.py]  게임별 리뷰 수 집계는 SQL GROUP BY 직접 쿼리로 전환 (전체 메타데이터 메모리 적재 방식 폐기)
```

## 코드 변경 사항

| 파일 | 변경 내용 |
|---|---|
| `src/config.py` | `VECTOR_DIR`(Chroma 경로) 제거. `DATABASE_URL`, `DATABASE_URL_DIRECT`를 env에서 로드하는 상수 추가 |
| `src/rag/index.py` | `Chroma` → `PGVector`로 교체. `get_vectorstore()` / `build_index()` 인터페이스는 유지해 호출부(`qa.py`) 영향 최소화 |
| `src/rag/qa.py` | 변경 없음 — `as_retriever(search_kwargs={"filter": {...}})` 인터페이스를 PGVector도 동일하게 지원 |
| `app.py` | `get_collection()`/`get_game_counts()`를 Chroma 전체 메타데이터 로드 방식에서 `SELECT app_name, COUNT(*) ... GROUP BY app_name HAVING COUNT(*) >= 20` 형태의 직접 SQL 집계로 교체 |
| `requirements.txt` | `chromadb`, `langchain-chroma` 제거. `langchain-postgres`, `psycopg[binary]` 추가 |
| `notebooks/06_build_index.ipynb` | 전면 교체: 80만 건 샘플링 → 정제 → 배치 임베딩 → Supabase 배치 insert → HNSW 인덱스 생성 |
| `chroma_store/` | 더 이상 필요 없어 삭제 대상 (단, git 반영은 사용자 지시 시까지 보류) |
| `.gitignore` | `!chroma_store/**` 예외 규칙 제거 (Chroma를 더 이상 git에 실을 필요 없음) |

## 환경변수

API 키는 불필요하다(직접 Postgres 연결 방식이므로 Supabase REST/anon 키 사용 안 함). `.env`(로컬 전용, gitignore 대상)에 아래 두 값만 추가한다:

```
DATABASE_URL=postgresql://postgres.[project-ref]:[비밀번호]@aws-0-[region].pooler.supabase.com:6543/postgres   # 앱 런타임용 (Transaction pooler)
DATABASE_URL_DIRECT=postgresql://postgres:[비밀번호]@db.[project-ref].supabase.co:5432/postgres              # 노트북 배치 적재·인덱스 생성용 (직접 연결)
```

- `DATABASE_URL`(6543, pooler): `app.py`/`qa.py`의 실시간 조회용. Streamlit Cloud 배포 시 Secrets에도 동일하게 등록.
- `DATABASE_URL_DIRECT`(5432, 직접 연결): `06_build_index.ipynb`의 대량 insert + `CREATE INDEX`(DDL) 실행 전용 — pooler보다 직접 연결이 안정적.
- Supabase 대시보드 → Project Settings → Database → Connection string에서 확인.

## 롤아웃 순서

1. `requirements.txt` 갱신
2. `src/config.py`, `src/rag/index.py`, `src/rag/qa.py`, `app.py` 수정
3. `.env`에 `DATABASE_URL`, `DATABASE_URL_DIRECT` 입력 (해당 시점에 사용자에게 안내)
4. `notebooks/06_build_index.ipynb`을 새 로직으로 교체 — 실행은 사용자가 직접 수행(기존 노트북 실행 정책 유지)
5. 로컬 `streamlit run app.py`로 게임 목록 조회 / RAG Q&A 동작 검증
6. 검증 통과 후 `chroma_store/` 삭제 + git 커밋 (사용자 명시적 지시 시에만)

## 리스크 / 트레이드오프

- **임베딩 소요 시간**: 80만 건 로컬 배치 임베딩은 상당한 시간이 걸릴 수 있음 — 중간 체크포인트/재개 로직을 노트북에 포함해 중단 시 재시작 비용을 줄인다.
- **Supabase 저장 용량**: 약 9~10GB 예상. 무료 티어(500MB)로는 부족하므로 Pro 이상 티어 필요(사용자 확인 완료, 이미 준비된 프로젝트 사용).
- **HNSW 인덱스 빌드 시간/메모리**: 80만 건 규모에서는 감내 가능한 수준으로 예상되나, 실제 적재 후 빌드 시간을 실측해 필요시 `ivfflat`으로 전환 검토.
- **게임별 집계 성능**: 기존 Chroma 방식(전체 메타데이터 메모리 로드)은 규모가 커지면 비효율적이므로 SQL 집계로 전환 — 이 부분은 이번 마이그레이션에서 함께 고친다(연관된 기존 코드 문제 개선).
