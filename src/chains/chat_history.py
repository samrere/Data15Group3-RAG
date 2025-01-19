from langchain_core.chat_history import InMemoryChatMessageHistory

history = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Get session's history by id"""
    if session_id not in history:
        history[session_id] = InMemoryChatMessageHistory()
    return history[session_id]