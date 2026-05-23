import yaml
from pathlib import Path


def load_config(name: str) -> dict:
    path = Path(__file__).parent / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
