import logging
import tempfile
from typing import Optional, Dict, List

import PyPDF2
import streamlit as st
from langchain_core.documents import Document

from config.config import get_config
from core.vectorstore import VectorStoreRepository
from chains.job_chain import JobSearchChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialise_session_state():
    """Initialise session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False


def setup_vectorstore() -> Optional[VectorStoreRepository]:
    """Setup vector store connection"""
    try:
        with st.spinner("Connecting to vector store..."):
            vector_store = VectorStoreRepository()
            vector_store.initialise()
            return vector_store
    except Exception as e:
        st.error(f"Error connecting to vector store: {str(e)}")
        return None


def read_resume_file(file) -> Optional[str]:
    """Read uploaded resume file"""
    try:
        with st.spinner("Reading resume..."):
            # Read PDF
            pdf_reader = PyPDF2.PdfReader(file)
            resume_text = ""
            for page in pdf_reader.pages:
                resume_text += page.extract_text()

            # Verify content
            if not resume_text.strip():
                st.error("Could not extract text from the PDF. Please check if the file is readable.")
                return None

            return resume_text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return None


def create_filter_dict(
    location: str,
    workplace: str,
    employment: str,
    industries: List[str],
    functions: List[str],
) -> Dict[str, str]:
    """Create filter dictionary for vector search"""
    filters = {}

    if location:
        filters["location"] = location
    if workplace != "Any":
        filters["workplace_type"] = workplace
    if employment != "Any":
        filters["employment_type"] = employment
    if industries:
        filters["industries"] = industries
    if functions:
        filters["job_functions"] = functions

    return filters


def main():
    # Get config
    config = get_config()

    # Set page config
    st.set_page_config(
        page_title=config.app_page_title,
        page_icon=config.app_page_icon,
        layout="wide"
    )

    # Initialise session state
    initialise_session_state()

    # Main title
    st.title("Jobs AI - Your Career Assistant")

    # Sidebar for resume upload, position and filters
    with st.sidebar:
        st.title("Resume Analysis")

        # Resume upload and position
        resume_file = st.file_uploader("Upload your resume", type=["txt", "pdf", "docx"])
        desired_position = st.text_input("Desired Position")

        # Add divider
        st.divider()

        # Search Filters section
        st.subheader("Search Filters")

        # Location filter
        location = st.text_input("Location", placeholder="e.g., Sydney, Melbourne")

        # Workplace type filter
        workplace_type = st.selectbox(
            "Workplace Type",
            options=["Any", "Remote", "Hybrid", "On-site"],
            index=0
        )

        # Employment type filter
        employment_type = st.selectbox(
            "Employment Type",
            options=[
                "Any",
                "Full-time",
                "Part-time",
                "Contract",
                "Temporary",
                "Internship",
            ],
            index=0
        )

        # Industry and function selection
        st.subheader("Industry Preferences")
        industry_options = ["Technology", "Finance", "Healthcare", "Education"]  # Example
        selected_industries = st.multiselect("Industries", options=industry_options)

        st.subheader("Job Function")
        function_options = ["Engineering", "Data Science", "Product", "Sales"]  # Example
        selected_functions = st.multiselect("Functions", options=function_options)

        if resume_file and desired_position:
            resume_text = read_resume_file(resume_file)
            if resume_text:
                if st.button("Analyze Resume"):
                    with st.spinner("Analyzing resume and finding matches..."):
                        # Set up vector store
                        vector_store = setup_vectorstore()
                        if vector_store:
                            # Create filter dictionary
                            filters = create_filter_dict(
                                location=location,
                                workplace=workplace_type,
                                employment=employment_type,
                                industries=selected_industries,
                                functions=selected_functions
                            )

                            try:
                                # Get retriever with filters
                                retriever = vector_store.get_retriever()
                                if filters:
                                    st.sidebar.write("Applied filters:", filters)
                                    retriever.search_kwargs["filter"] = filters

                                # Create chain
                                chain = JobSearchChain(retriever)

                                # Perform analysis
                                result = chain.analyse_resume(
                                    resume_text=resume_text,
                                    position=desired_position
                                )

                                # Store chain and mark analysis complete
                                st.session_state.chain = chain
                                st.session_state.analysis_complete = True

                                # Display analysis
                                st.subheader("Analysis")
                                st.write(result["analysis"])

                                # Display matched jobs
                                st.subheader("Top Matching Jobs")
                                for job_match in result["matched_jobs"]:
                                    job = job_match["job"]
                                    score = job_match["match_score"]
                                    
                                    with st.expander(f"{job['title']} at {job['company']} (Match: {score:.2%})"):
                                        st.write(f"Location: {job['location']}")
                                        st.write(f"Type: {job['employment_type']}")
                                        st.write(f"Skills: {', '.join(job['skills'])}")
                                        st.write("Description:", job['description'])

                                # Add analysis to chat
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": result["analysis"]}
                                )

                                st.success("Analysis complete! Feel free to ask questions about the matches.")
                            except Exception as e:
                                st.error(f"Analysis failed: {str(e)}")

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Chat input
    if st.session_state.analysis_complete:
        if prompt := st.chat_input("Ask a follow-up question about the jobs or resume advice"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Force refresh to show user message immediately
            st.rerun()

        # Check if we need to process the last user message
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            try:
                with st.spinner("Thinking..."):
                    # Get last user message
                    last_message = st.session_state.messages[-1]["content"]

                    # Get response from chain
                    response = st.session_state.chain.chat(last_message)

                    # Add assistant response
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response["answer"]}
                    )

                    # Force refresh to show assistant response
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
    else:
        st.info("Please upload your resume and complete the analysis to start chatting.")


if __name__ == "__main__":
    main()