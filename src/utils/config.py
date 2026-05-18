import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def load_config(file_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load entire config from TOML file.

    Args:
        file_path: Path to config.toml. If None, uses default location.

    Returns:
        Dictionary with full config.
    """
    if file_path is None:
        file_path = PROJECT_ROOT / "config.toml"
    with open(file_path, "rb") as f:
        return tomllib.load(f)

def get_config(file_path: Optional[Path] = None) -> SimpleNamespace:
    """Load config from TOML file and convert to a nested SimpleNamespace.

    Args:
        file_path: Path to config.toml. If None, uses default location.

    Returns:
        SimpleNamespace with config values accessible as attributes.
    """
    data = load_config(file_path)
    return _dict_to_ns(data)


def _dict_to_ns(data: Dict[str, Any]) -> SimpleNamespace:
    """Recursively convert a dictionary to a SimpleNamespace.

    Args:
        data: Dictionary to convert. Nested dicts are converted to nested SimpleNamespace objects.

    Returns:
        SimpleNamespace with keys as attributes and nested dicts recursively converted.
    """
    ns = SimpleNamespace()
    for key, value in data.items():
        setattr(ns, key, _dict_to_ns(value) if isinstance(value, dict) else value)
    return ns

# -----------------------------------------------

class GraphConfig(NamedTuple):
    """Configuration for Graph RAG implementation."""
    uri: str
    user: str
    password: str
    embedding_model: str
    bert_model_name: str

    @classmethod
    def from_dict(cls, data: dict) -> "GraphConfig":
        """Create GraphConfig from a dictionary.

        Args:
            data: Dictionary containing neo4j, embedding and extraction config keys.

        Returns:
            GraphConfig instance.
        """
        return cls(
            uri=data.get("neo4j", {}).get("uri", ""),
            user=data.get("neo4j", {}).get("user", ""),
            password=data.get("neo4j", {}).get("password", ""),
            embedding_model=data.get("embedding", {}).get("model", "BAAI/bge-small-en-v1.5"),
            bert_model_name=data.get("extraction", {}).get("bert_model", "bert-base-uncased"),
        )

    @classmethod
    def load(cls, file_path: Optional[Path] = None) -> "GraphConfig":
        """Load GraphConfig from config.toml file.

        Args:
            file_path: Path to config.toml. If None, uses default location.

        Returns:
            GraphConfig instance.
        """
        data = load_config(file_path)
        return cls.from_dict(data)


class VectorConfig(NamedTuple):
    """Configuration for Vector RAG implementation."""
    host: str
    port: int
    model: str
    dimension: int

    @classmethod
    def from_dict(cls, data: dict) -> "VectorConfig":
        """Create VectorConfig from a dictionary.

        Args:
            data: Dictionary containing config keys (qdrant host/port, embedding model/dimension).

        Returns:
            VectorConfig instance.
        """
        return cls(
            host=data.get("qdrant", {}).get("host", ""),
            port=data.get("qdrant", {}).get("port", 0),
            model=data.get("embedding", {}).get("model", ""),
            dimension=data.get("embedding", {}).get("dimension", 0),
        )

    @classmethod
    def load(cls, file_path: Optional[Path] = None) -> "VectorConfig":
        """Load VectorConfig from config.toml file.

        Args:
            file_path: Path to config.toml. If None, uses default location.

        Returns:
            VectorConfig instance.
        """
        data = load_config(file_path)
        return cls.from_dict(data)
