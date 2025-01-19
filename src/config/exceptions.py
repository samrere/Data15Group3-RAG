class JobsAIException(Exception):
    """Base exception for the application"""
    pass


class ConfigurationError(JobsAIException):
    """Raised when there's a configuration error"""
    pass


class VectorStoreError(JobsAIException):
    """Raised when there's an error with vector store operations"""
    pass


class ChainError(JobsAIException):
    """Raised when there's an error with chain execution"""
    pass