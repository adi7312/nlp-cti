import tomllib
from pathlib import Path

def load_config(file_path: Path | None = None):
    if file_path is None:
        file_path = Path(__file__).parent.parent / "config.toml"

    with open(file_path, "rb") as f:
        return tomllib.load(f)
