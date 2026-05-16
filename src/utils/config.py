import tomllib
from pathlib import Path
from typing import NamedTuple, Optional, Dict, Any


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


def load_config(file_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load entire config from TOML file.

    Args:
        file_path: Path to config.toml. If None, uses default location.

    Returns:
        Dictionary with full config.
    """
    if file_path is None:
        file_path = Path(__file__).parent.parent / "config.toml"

    with open(file_path, "rb") as f:
        return tomllib.load(f)
