from sqlalchemy import Column, ForeignKey, Integer, String, Float
from pgvector.sqlalchemy import Vector
from database import Base

class JobPosting(Base):
    __tablename__ = "job_postings"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    job_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, index=True, nullable=False)
    desc_raw = Column(String, nullable=False)
    desc_sbert = Column(String, nullable=True)
    desc_tfidf = Column(String, nullable=True)
    formatted_work_type = Column(String, nullable=False)
    company = Column(String, nullable=True)
    formatted_experience_level = Column(String, nullable=True)

class JobEmbedding(Base):
    __tablename__ = "job_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    embedding = Column(Vector(384), nullable=False)
    model_version = Column(String, nullable=False)
    job_posting_id = Column(Integer, ForeignKey("job_postings.id", ondelete="CASCADE"), nullable=False, index=True)