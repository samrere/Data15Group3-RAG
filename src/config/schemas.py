from pydantic import BaseModel, HttpUrl
from typing import List, Optional


class Job(BaseModel):
    """Job model"""

    job_id: str
    title: str
    company: str
    location: str
    employment_type: str
    seniority_level: str = ""
    industries: List[str]
    job_functions: List[str]
    workplace_type: str
    description: str
    skills: List[str]
    job_url: HttpUrl
    reposted: Optional[bool] = None
    posted_time: int
    expire_time: int
    apply_url: Optional[HttpUrl] = None


class VectorSearchResult(BaseModel):
    """Vector search result model"""

    job: Job
    score: float