from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRAG(ABC):
    """Abstract base class for RAG (Retrieval-Augmented Generation) implementations.

    Subclasses must implement ingestion, search, and storage initialization methods.
    """

    @abstractmethod
    def ingest(self, data: Any, **kwargs: Dict[str, Any]) -> None:
        """Ingest data into the storage backend.

        Args:
            data: The data to ingest (e.g., PDF paths, documents, relations).
            **kwargs: Additional ingestion parameters (e.g., collection name, strategy).
        """
        pass

    @abstractmethod
    def search(self, query: str, **kwargs: Dict[str, Any]) -> List[Any]:
        """Search the storage backend for relevant information.

        Args:
            query: The search query string.
            **kwargs: Additional search parameters (e.g., collection name, limit).

        Returns:
            List of search results.
        """
        pass

    @abstractmethod
    def init_storage(self, name: str, **kwargs: Dict[str, Any]) -> None:
        """Initialize the storage backend (e.g., create collection, ensure schema).

        Args:
            name: Name of the storage collection/database.
            **kwargs: Additional initialization parameters.
        """
        pass
