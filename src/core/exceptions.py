class PriceIsRightBaseException(Exception):
    """Base exception for the application."""
    # pass

class AgentExecutionError(PriceIsRightBaseException):
    """Raised when an agent fails to complete its task."""
    # pass

class ModelInferenceError(PriceIsRightBaseException):
    """Raised when LLM or DNN inference fails."""
    # pass

class VectorStoreError(PriceIsRightBaseException):
    """Raised when ChromaDB operations fail."""
    # pass

class ExtractionError(PriceIsRightBaseException):
    """Raised when the scanner fails to parse deal information."""
    # pass