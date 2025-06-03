# my_local_assistant/__init__.py
from .core.llm_handler import LLMHandler
from .core.vector_store import VectorStoreHandler
from .core.assistant import RAGAssistant

__version__ = "0.1.0" # Keep in sync with pyproject.toml

__all__ = ["LLMHandler", "VectorStoreHandler", "RAGAssistant"]