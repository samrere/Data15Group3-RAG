from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_analysis_prompt() -> ChatPromptTemplate:
    """Create prompt for resume analysis"""

    template = """Analyze the provided resume for job matching. Consider:

    Resume:
    {resume}
    
    Desired Position:
    {position}
    
    Available Jobs:
    {context}
    
    Provide a clear analysis covering:
    1. Overview of candidate's background and job match potential
    2. Key skills and experience relevant to the matched positions
    3. Specific recommendations for top 3 most promising matches
    4. Suggested resume updates for target positions
    5. Cover letter focus points for applications
    
    Keep the response practical and actionable, focusing on helping the candidate succeed in their applications."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", template),
            ("human", "Please analyze my resume for these positions."),
        ]
    )


def create_chat_prompt() -> ChatPromptTemplate:
    """Create prompt for follow-up questions"""

    template = """You are a helpful career coach. Answer questions based on the prior analysis and job matches.
    
    Initial Analysis:
    {initial_analysis}

    Related Job Information:
    {context}

    Provide specific, actionable advice related to the job search."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )


def create_condense_prompt() -> ChatPromptTemplate:
    """Create prompt for condensing follow-up questions with resume context"""

    condense_template = """Given the user's resume background and chat history, formulate a standalone question that captures the full context of their current question.

    Important aspects to maintain:
    1. User's experience level and background
    2. Previously discussed career goals
    3. Specific job interests identified
    4. Skill development priorities

    DO NOT answer the question - only reformulate it if needed based on chat history and resume context, or return it unchanged if already clear.

    RESUME ANALYSIS CONTEXT:
    {initial_analysis}
    
    CHAT HISTORY:
    {chat_history}
    
    CURRENT QUESTION: {input}
    """

    return ChatPromptTemplate.from_messages(
        [
            ("system", condense_template),
        ]
    )
