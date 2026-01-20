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

class EmbeddingBase(BaseModel):
    embedding: List[float]
    model_version: str
    job_posting_id: int

class PostingBase(BaseModel):
    job_id: str
    title: str
    desc_raw: str
    desc_sbert: Optional[str] = None
    desc_tfidf: Optional[str] = None 
    formatted_work_type: str
    company: Optional[str] = None
    formatted_experience_level: Optional[str] = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

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
    