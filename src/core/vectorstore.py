import logging
from typing import Dict, List, Optional, Union

import pinecone
from pinecone import Index
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import VectorStoreRetriever

from config.config import get_config
from config.exceptions import VectorStoreError
from config.schemas import Job, VectorSearchResult

logger = logging.getLogger(__name__)


class VectorStoreRepository:
    """Managing vector store operations"""

    def __init__(self):
        self.config = get_config()
        self.embeddings = OpenAIEmbeddings(
            api_key=self.config.openai_api_key,
            model=self.config.openai_embedding_model
        )
        self.vectorstore = None
        self.pc = pinecone.Pinecone(api_key=self.config.pinecone_api_key)

    def initialise(self, documents: Optional[List[Document]] = None) -> None:
        """Initialise vector store"""
        if not self.config.pinecone_api_key:
            raise VectorStoreError("Pinecone API key not configured")

        try:
            index_name = self.config.pinecone_index_name

            # Try to connect to existing index
            try:
                logger.info(f"Attempting to connect to index: {index_name}")
                index = self.pc.Index(index_name)
                self.vectorstore = PineconeVectorStore(
                    index=index,
                    embedding=self.embeddings
                )
                logger.info("Successfully connected to existing index")
                return

            except Exception as e:
                logger.error(f"Could not connect to existing index: {str(e)}")
                raise VectorStoreError(f"Failed to connect to index: {str(e)}")

        except Exception as e:
            logger.error(f"Error initialising vector store: {str(e)}")
            raise VectorStoreError(f"Failed to initialise vector store: {str(e)}")

    def get_retriever(self) -> VectorStoreRetriever:
        """Get retriever with current configuration"""
        if self.vectorstore is None:
            raise VectorStoreError("Vector store not initialized")
        return self.vectorstore.as_retriever()

    def _prepare_filter_conditions(
        self,
        filter_dict: Dict[str, Union[str, List[str]]]
    ) -> Dict:
        """Convert dictionary to Pinecone filter format"""
        filter_conditions = {}
        for key, value in filter_dict.items():
            if isinstance(value, list):
                filter_conditions[key] = {"$in": value}
            else:
                filter_conditions[key] = {"$eq": value}
        return filter_conditions

    def search_jobs(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Union[str, List[str]]]] = None,
        k: int = 10
    ) -> List[VectorSearchResult]:
        """Search for relevant jobs with similarity scores"""
        if self.vectorstore is None:
            raise VectorStoreError("Vector store not initialized")

        try:
            search_params = {"k": k}
            if filter_dict:
                search_params["filter"] = self._prepare_filter_conditions(filter_dict)

            results = self.vectorstore.similarity_search_with_score(
                query=query,
                **search_params
            )

            return [
                VectorSearchResult(
                    job=Job(**doc.metadata),
                    score=score
                )
                for doc, score in results
            ]
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise VectorStoreError(f"Failed to search vector store: {str(e)}")