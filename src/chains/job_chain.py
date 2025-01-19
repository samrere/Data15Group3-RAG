import logging
from typing import Dict, Any, List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever

from chains.chat_history import get_session_history
from chains.prompts import (
    create_analysis_prompt,
    create_chat_prompt,
    create_condense_prompt,
)
from config.exceptions import ChainError
from config.schemas import Job, VectorSearchResult
from core.llm import LLM

logger = logging.getLogger(__name__)


class JobSearchChain:
    """Chain for job search queries with chat history"""

    def __init__(self, retriever: VectorStoreRetriever):
        self.llm = LLM()
        self.retriever = retriever
        self.chain = self._create_chain()
        self.initial_analysis = None
        self.matched_jobs = None

    def analyse_resume(self, resume_text: str, position: str) -> dict:
        """Analyze resume and find matching jobs"""
        try:
            # Get relevant documents using retriever
            docs = self.retriever.get_relevant_documents(query=resume_text)

            # Convert to our format (using 1.0 - rank/total as a simple scoring mechanism)
            total_docs = len(docs)
            matched_jobs = [
                VectorSearchResult(
                    job=Job(**doc.metadata),
                    score=1.0 - (idx / total_docs),  # Simple scoring mechanism
                )
                for idx, doc in enumerate(docs)
            ]

            # Store matched jobs for context
            self.matched_jobs = matched_jobs

            # Create analysis chain
            analysis_prompt = create_analysis_prompt()
            analysis_chain = create_stuff_documents_chain(
                llm=self.llm.llm, prompt=analysis_prompt
            )

            # Generate analysis
            analysis = analysis_chain.invoke(
                {
                    "resume": resume_text,
                    "position": position,
                    "context": docs,
                }
            )

            # Store for chat context
            self.initial_analysis = analysis

            return {
                "analysis": analysis,
                "matched_jobs": [
                    {"job": result.job.dict(), "match_score": result.score}
                    for result in matched_jobs
                ],
            }

        except Exception as e:
            logger.error(f"Error in resume analysis: {str(e)}")
            raise ChainError(f"Failed to analyze resume: {str(e)}")

    def _create_chain(self) -> RunnableWithMessageHistory:
        """Create the chat chain with history awareness"""
        try:
            # Create history-aware retriever for follow-up questions
            condense_prompt = create_condense_prompt()
            history_aware_retriever = create_history_aware_retriever(
                llm=self.llm.llm, retriever=self.retriever, prompt=condense_prompt
            )

            # Create QA chain for job-specific responses
            qa_prompt = create_chat_prompt()
            qa_chain = create_stuff_documents_chain(llm=self.llm.llm, prompt=qa_prompt)

            # Combine retriever and QA into single chain
            retrieval_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

            # Wrap with message history
            return RunnableWithMessageHistory(
                retrieval_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )

        except Exception as e:
            logger.error(f"Error creating job search chain: {str(e)}")
            raise ChainError(f"Failed to create job search chain: {str(e)}")

    def chat(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """Process follow-up question"""
        if not self.initial_analysis:
            raise ChainError("Resume must be analysed before starting chat")

        logger.debug(f"Processing chat question: {question}")

        try:
            response = self.chain.invoke(
                {"input": question, "initial_analysis": self.initial_analysis},
                config={"configurable": {"session_id": session_id}},
            )

            return {
                "answer": response["answer"],
                "context": response.get("context", []),
                "source_documents": response.get("source_documents", []),
            }

        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return {
                "answer": "I apologise, but I encountered an error. Please try again.",
                "context": [],
                "source_documents": [],
            }
