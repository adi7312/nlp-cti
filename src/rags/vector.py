from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from typing import List, Dict, Any, cast

from src.rags.base import BaseRAG
from src.utils.config import VectorConfig
from src.utils.chunk_strategies import chunk_text


class VectorRAG(BaseRAG):
    """Vector RAG implementation using Qdrant as the vector store."""

    def __init__(self, config: VectorConfig):
        """Initialize the VectorRAG with Qdrant client and embedding model.

        Args:
            config: VectorConfig instance with host, port, model, and dimension.
        """
        self.qdrant_client = QdrantClient(host=config.host, port=config.port)
        self.embedding_model = SentenceTransformer(config.model)
        self.lc_embedding_model = HuggingFaceEmbeddings(model_name=config.model)
        self.vector_size = config.dimension

    def init_storage(self, name: str, **kwargs: Dict[str, Any]) -> None:
        """Ensure the Qdrant collection exists with the correct vector dimensions.

        Args:
            name: Name of the collection to initialize.
            **kwargs: Additional parameters (unused).
        """
        collections = self.qdrant_client.get_collections().collections
        if not any(col.name == name for col in collections):
            print(f"Creating Qdrant collection: {name}")
            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
        else:
            print(f"Collection {name} already exists.")

    def ingest(self, data: List[str], **kwargs: Dict[str, Any]) -> None:
        """Load PDFs, chunk them, vectorize them, and upload to Qdrant.

        Args:
            data: List of PDF file paths to ingest.
            **kwargs: Additional parameters including:
                - collection_name: Name of the collection to ingest to (default: "cti_reports")
                - strategy: Chunking strategy (default: "sliding_window")
        """
        collection_name = cast(str, kwargs.get("collection_name", "cti_reports"))
        strategy = cast(str, kwargs.get("strategy", "sliding_window"))

        collections = self.qdrant_client.get_collections().collections
        if any(col.name == collection_name for col in collections):
            print(f"Clearing old data from {collection_name}...")
            self.qdrant_client.delete_collection(collection_name)

        self.init_storage(collection_name)

        all_chunks = []
        for path in data:
            print(f"Loading {path}...")
            loader = PyPDFLoader(path)
            docs = loader.load()
            chunks = chunk_text(docs, strategy=strategy, embedding_model=self.lc_embedding_model)
            all_chunks.extend(chunks)

        print(f"Total chunks created using '{strategy}' strategy: {len(all_chunks)}")

        points = []
        for chunk in all_chunks:
            text = chunk.page_content
            metadata = chunk.metadata
            vector = self.embedding_model.encode(text).tolist()
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, "source": metadata.get("source", "Unknown")},
            )
            points.append(point)

        if points:
            print("Uploading vectors to Qdrant...")
            self.qdrant_client.upsert(collection_name=collection_name, points=points)
            print("Upload complete!")

    def search(self, query: str, **kwargs: Dict[str, Any]) -> List[str]:
        """Semantic search within Qdrant.

        Args:
            query: Search query string.
            **kwargs: Additional parameters including:
                - collection_name: Name of the collection to search (default: "cti_reports")

        Returns:
            List of matching text chunks.
        """
        collection_name = cast(str, kwargs.get("collection_name", "cti_reports"))
        query_vector = self.embedding_model.encode(query).tolist()
        results = self.qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=2,
            with_payload=True
        )
        return [point.payload['text'] for point in results.points]
