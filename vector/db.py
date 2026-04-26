from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SpacyTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, QDRANT_HOST, QDRANT_PORT

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
lc_embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
VECTOR_SIZE = EMBEDDING_DIMENSION  # Configured in config.py


def init_qdrant_collection(collection_name="cti_reports"):
    """Ensures the Qdrant collection exists with the correct vector dimensions."""
    collections = qdrant_client.get_collections().collections
    if not any(col.name == collection_name for col in collections):
        print(f"Creating Qdrant collection: {collection_name}")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        print(f"Collection {collection_name} already exists.")


def chunk_text(docs, strategy="sliding_window"):
    """Applies chunking strategy to loaded documents."""
    if strategy == "fixed":
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    elif strategy == "sliding_window":
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    elif strategy == "sentence":
        splitter = RecursiveCharacterTextSplitter(
            separators=[".\n", "?\n", "!\n", ". ", "? ", "! "],
            chunk_size=300,
            chunk_overlap=0,
        )
    elif strategy == "semantic":
        splitter = SemanticChunker(lc_embedding_model)
    else:
        raise ValueError("Unknown chunking strategy")

    return splitter.split_documents(docs)


def ingest_pdfs_to_qdrant(pdf_paths, strategy="sliding_window"):
    """Loads PDFs, chunks them, vectorizes them, and uploads to Qdrant."""
    collection_name = f"cti_reports_{strategy}"

    collections = qdrant_client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        print(f"Clearing old data from {collection_name}...")
        qdrant_client.delete_collection(collection_name)

    init_qdrant_collection(collection_name)

    all_chunks = []
    for path in pdf_paths:
        print(f"Loading {path}...")
        loader = PyPDFLoader(path)
        docs = loader.load()
        chunks = chunk_text(docs, strategy=strategy)
        all_chunks.extend(chunks)

    print(f"Total chunks created using '{strategy}' strategy: {len(all_chunks)}")

    points = []
    for chunk in all_chunks:
        text = chunk.page_content
        metadata = chunk.metadata
        vector = embedding_model.encode(text).tolist()
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"text": text, "source": metadata.get("source", "Unknown")},
        )
        points.append(point)

    if points:
        print("Uploading vectors to Qdrant...")
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print("Upload complete!")


def search_vector(query: str, collection_name: str = "cti_reports_sliding_window"):
    """Semantic search within Qdrant."""
    query_vector = embedding_model.encode(query).tolist()
    results = qdrant_client.search(collection_name=collection_name, query_vector=query_vector, limit=2)
    return [hit.payload['text'] for hit in results]
