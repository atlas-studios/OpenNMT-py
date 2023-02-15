from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Get env variables
from dotenv import load_dotenv
import os
load_dotenv()
SQLALCHEMY_DATABASE_URL = os.getenv("POSTGRES_URL")

engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=100, max_overflow=0, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except:
        db.close()