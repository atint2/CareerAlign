from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Annotated, Optional, Dict, Any
from backend import models
from backend.database import engine, SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.matcher.hybrid_matcher import hybrid_match, downstream_match

# Initialize FastAPI app
app = FastAPI()

# Create vector extension in PostgreSQL if it doesn't exist
with engine.connect() as connection:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    connection.commit()

# Create database tables based on the defined SQLAlchemy models
models.Base.metadata.create_all(bind=engine)

# Enable RLS on all tables
tables = ["job_postings", "job_embeddings_sbert", "job_embeddings_tfidf", "reduced_job_embeddings", "clusters", "cluster_embeddings_tfidf", "cluster_embeddings_sbert", "resumes"]
with engine.connect() as connection:
    for table in tables:
        connection.execute(text(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;"))
    connection.commit()

# Endpoint for health check
@app.get('/api/ping')
async def ping():
    return {'message': 'Hello from Python backend!'}

# Pydantic models for request and response validation
class PostingBase(BaseModel):
    job_id: str
    title: str
    desc_raw: str
    desc_sbert: Optional[str] = None
    desc_tfidf: Optional[str] = None 
    formatted_work_type: str
    company: Optional[str] = None
    formatted_experience_level: Optional[str] = None
    cluster_id: Optional[int] = None

class SBERTEmbeddingBase(BaseModel):
    embedding: List[float]
    model_version: str
    job_posting_id: int

class TFIDFEmbeddingBase(BaseModel):
    embedding: List[float]
    job_posting_id: int

class ReducedEmbeddingBase(BaseModel):
    reduced_embedding: List[float]
    model_version: str
    job_embedding_id: int
    reduction_method: str

class ClusterBase(BaseModel):
    cluster_id: int
    title: Optional[str] = None
    general_job_desc_raw: Optional[str] = None
    general_job_desc_sbert: Optional[str] = None
    general_job_desc_tfidf: Optional[str] = None
    num_postings: int  

# For TF-IDF and SBERT cluster embeddings
class ClusterEmbeddingBase(BaseModel):
    embedding: List[float]
    cluster_id: int

class ResumeBase(BaseModel):
    resume_id: str
    content_raw: str
    content_sbert: Optional[str] = None
    content_tfidf: Optional[str] = None

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

# Endpoint to retrieve all job postings
@app.get('/api/postings/', response_model=List[PostingBase])
async def get_job_postings(db: db_dependency):
    postings = db.query(models.JobPosting).all()
    return postings
    
# Endpoint to retrieve all job embeddings
@app.get('/api/embeddings/', response_model=List[SBERTEmbeddingBase])
async def get_job_embeddings(db: db_dependency):
    embeddings = db.query(models.JobEmbeddingSBERT).all()
    return embeddings

# Endpoint to create a new job embedding
@app.post('/api/embeddings/')
async def create_job_posting_embedding(JobEmbedding: SBERTEmbeddingBase, db: db_dependency):
    db_embedding = models.JobEmbeddingSBERT(embedding=JobEmbedding.embedding,
                                       model_version=JobEmbedding.model_version,
                                       job_posting_id=JobEmbedding.job_posting_id)
    db.add(db_embedding)
    db.commit()
    db.refresh(db_embedding)
    return db_embedding

# Endpoint to create a new reduced job embedding
@app.post('/api/reduced-embeddings/')
async def create_reduced_embedding(ReducedEmbedding: ReducedEmbeddingBase, db: db_dependency):
    db_reduced_embedding = models.ReducedEmbedding(reduced_embedding=ReducedEmbedding.reduced_embedding,
                                                   model_version=ReducedEmbedding.model_version,
                                                   job_embedding_id=ReducedEmbedding.job_embedding_id,
                                                   reduction_method=ReducedEmbedding.reduction_method)
    db.add(db_reduced_embedding)
    db.commit()
    db.refresh(db_reduced_embedding)
    return db_reduced_embedding

class ResumeMatchRequest(BaseModel):
    resume_text: str
    job_desc: Optional[str] = None

@app.post("/api/hybrid-match-resume/")
async def hybrid_match_resume_endpoint(
    request: ResumeMatchRequest,
    db: db_dependency
):
    try:
        results = hybrid_match(request.resume_text, request.job_desc, db)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DownstreamMatchRequest(BaseModel):
    resume_text: str
    hybrid_matches: List[Dict[str, Any]]

@app.post("/api/downstream-match-resume/")
async def downstream_match_resume(
    request: DownstreamMatchRequest,
    db: db_dependency
):
    try:
        results = downstream_match(request.resume_text, request.hybrid_matches, db)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))