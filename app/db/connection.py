import psycopg2
from app.core.settings import settings

def get_connection(database=None):
    return psycopg2.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=database or settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD
    )