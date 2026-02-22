from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Annotated, Optional
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session

# Initialize FastAPI app
app = FastAPI()
models.Base.metadata.create_all(bind=engine) # Create database tables in PostgreSQL

# Set up CORS middleware to allow requests from the frontend
origins = [
    "http://localhost:5173"
]

# Add CORS middleware to the FastAPI application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all HTTP methods
    allow_headers=["*"], # Allow all headers
)

@app.get('/api/ping')
async def ping():
    return {'message': 'Hello from Python backend!'}

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

class EmbeddingBase(BaseModel):
    embedding: List[float]
    model_version: str
    job_posting_id: int

class ReducedEmbeddingBase(BaseModel):
    reduced_embedding: List[float]
    model_version: str
    job_embedding_id: int
    reduction_method: str

class ClusterBase(BaseModel):
    cluster_id: int
    cluster_desc: Optional[str] = None
    num_postings: int

class ResumeBase(BaseModel):
    filename: str
    content: str
    content_sbert: Optional[str] = None
    content_tfidf: Optional[str] = None

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

# Endpoint to create a new job posting
@app.post('/api/postings/')
async def create_job_posting(JobPosting: PostingBase, db: db_dependency):
    db_posting = models.JobPosting(job_id=JobPosting.job_id,
                                    title=JobPosting.title,
                                    desc_raw=JobPosting.desc_raw,
                                    desc_sbert=JobPosting.desc_sbert,
                                    desc_tfidf=JobPosting.desc_tfidf,
                                    formatted_work_type=JobPosting.formatted_work_type,
                                    company=JobPosting.company,
                                    formatted_experience_level=JobPosting.formatted_experience_level)
    db.add(db_posting)
    db.commit()
    db.refresh(db_posting)
    return db_posting
    
# Endpoint to retrieve all job embeddings
@app.get('/api/embeddings/', response_model=List[EmbeddingBase])
async def get_job_embeddings(db: db_dependency):
    embeddings = db.query(models.JobEmbedding).all()
    return embeddings

# Endpoint to create a new job embedding
@app.post('/api/embeddings/')
async def create_job_embedding(JobEmbedding: EmbeddingBase, db: db_dependency):
    db_embedding = models.JobEmbedding(embedding=JobEmbedding.embedding,
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