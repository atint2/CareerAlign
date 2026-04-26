from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

engine = None
SessionLocal = None

def init_db():
    global engine, SessionLocal

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is missing")

    engine = create_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize immediately on import so get_db works before lifespan runs
init_db()