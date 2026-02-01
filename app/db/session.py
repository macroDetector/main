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
    with get_connection(database="postgres") as conn:
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (settings.DB_NAME,)
            )
            if not cursor.fetchone():
                cursor.execute(f"CREATE DATABASE {settings.DB_NAME}")
                print(f"[DB] Database '{settings.DB_NAME}' created")

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

            ensure_columns(cursor, "mouse_points", {
                "event_type": "SMALLINT NOT NULL DEFAULT 0",   # 0=move,1=down,2=up
                "is_pressed": "SMALLINT NOT NULL DEFAULT 0",   # 상태
                "deltatime": 'FLOAT NOT NULL DEFAULT 0'
            })

        with conn.cursor() as cursor:
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro_mouse_points (
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
