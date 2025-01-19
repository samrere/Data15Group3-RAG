from typing import Any, Dict, List, Union
import logging
from langchain_core.documents import Document
from config.schemas import Job

logger = logging.getLogger(__name__)


class JobDocumentLoader:
    """Custom document loader for Job objects with Pinecone-compatible metadata"""

    def _sanitise_metadata_value(
        self, value: Any
    ) -> Union[str, float, bool, List[str]]:
        """Sanitise a single metadata value to Pinecone-compatible type"""
        if isinstance(value, (str, bool)):
            return value
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, list):
            return [str(item) for item in value]
        else:
            # Convert any other types to string
            return str(value)

    def _create_metadata(
        self, job: Job
    ) -> Dict[str, Union[str, float, bool, List[str]]]:
        """Create Pinecone-compatible metadata dictionary from Job object"""
        try:
            raw_metadata = {
                "job_id": job.job_id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "employment_type": job.employment_type,
                "seniority_level": job.seniority_level,
                "industries": job.industries,
                "job_functions": job.job_functions,
                "workplace_type": job.workplace_type,
                "skills": job.skills,
                "job_url": str(job.job_url),
                "reposted": bool(job.reposted) if job.reposted is not None else False,
                "posted_time": float(job.posted_time),
                "expire_time": float(job.expire_time),
                "apply_url": str(job.apply_url) if job.apply_url else None,
            }

            # Sanitise all values to ensure Pinecone compatibility
            sanitised_metadata = {
                key: self._sanitise_metadata_value(value)
                for key, value in raw_metadata.items()
                if value is not None
            }

            return sanitised_metadata
        except Exception as e:
            logger.error(f"Error creating metadata for job: {str(e)}")
            logger.debug(f"Job data: {job}")
            raise ValueError(f"Failed to create metadata: {str(e)}")

    def load_jobs(self, jobs: List[Job]) -> List[Document]:
        """Convert Job objects to Documents with Pinecone-compatible metadata"""
        try:
            documents = []
            for job in jobs:
                # Create document with description as content and sanitised metadata
                doc = Document(
                    page_content=job.description,
                    metadata=self._create_metadata(job)
                )
                documents.append(doc)

            logger.debug(f"Converted {len(documents)} jobs to documents")
            return documents
        except Exception as e:
            logger.error(f"Error converting jobs to documents: {str(e)}")
            raise

    def process_test_job(self, job: Job) -> Document:
        """Convert a single Job to Document - for testing"""
        try:
            return Document(
                page_content=job.description,
                metadata=self._create_metadata(job)
            )
        except Exception as e:
            logger.error(f"Error converting test job to document: {str(e)}")
            raise