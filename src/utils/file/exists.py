from pathlib import Path


def path_exists(path: str) -> bool:
    return Path(path).exists()
