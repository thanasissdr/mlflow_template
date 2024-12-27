from pathlib import Path

from src.utils.file.exists import path_exists


def create_file(file_path: str) -> None:
    Path(file_path).touch()


def create_dir(dir_path: str, parents: bool = True, exist_ok: bool = False) -> None:
    Path(dir_path).mkdir(parents=parents, exist_ok=exist_ok)


def create_dir_if_not_exists(dir_path: str) -> None:
    if not path_exists(dir_path):
        create_dir(dir_path)


def create_file_file_if_not_exists(file_path: str) -> None:
    if not path_exists(file_path):
        create_file(file_path)
