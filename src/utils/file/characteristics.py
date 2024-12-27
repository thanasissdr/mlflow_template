from pathlib import Path


def get_file_extension(file_path: str) -> str:
    return Path(file_path).suffix


def get_file_path_without_extension(file_path: str) -> str:
    path = Path(file_path)
    return str(path.with_name(path.stem))


def get_file_size(file_path: str) -> int:
    return Path(file_path).stat().st_size


def get_dirname(path: str) -> str:
    return str(Path(path).parent)


def is_dir(path: str) -> bool:
    return Path(path).is_dir()
