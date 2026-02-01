from app.db.connection import get_connection
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from app.core.settings import settings

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def ensure_columns(cursor, table_name: str, required_columns: dict):
    cursor.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s
    """, (table_name,))

    existing_columns = {row[0] for row in cursor.fetchall()}

    for col, col_def in required_columns.items():
        if col not in existing_columns:
            cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {col} {col_def}"
            )
            print(f"[DB] {table_name}: column '{col}' added")


def init_db():
    # ---------- DB 생성 ----------
    conn = get_connection(database="postgres")
    
    try:
        # 2. 트랜잭션 블록에 들어가기 전에 AUTOCOMMIT을 설정합니다.
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (settings.DB_NAME,)
            )
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {settings.DB_NAME}")
                print(f"[DB] Database '{settings.DB_NAME}' created")
    finally:
        # 3. 작업이 끝나면 명시적으로 닫아줍니다.
        conn.close()

    # ---------- 테이블 & 컬럼 보정 ----------
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS mouse_points (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                deltatime Float NOT NULL DEFAULT 0.0
            )
            """)

        conn.commit()


# ---------- SQLAlchemy ----------
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
