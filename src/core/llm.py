from langchain_openai import ChatOpenAI
from config.config import get_config


class LLM:
    """Chat model wrapper"""

    def __init__(self):
        self.config = get_config()
        self.llm = ChatOpenAI(
            api_key=self.config.openai_api_key,
            model="gpt-4",
            temperature=0.1
        )