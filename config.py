# Global configuration constants
# Ensures consistency across the entire project

# Embedding Model Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384  # Output dimension of BAAI/bge-small-en-v1.5

# Vector Database Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# LLM Configuration
LOCAL_API_URL = "http://172.20.224.1:1337/v1"
LLM_MODEL_NAME = "gemma-4-E4B-it-Q8_0"

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "twoje_haslo"
