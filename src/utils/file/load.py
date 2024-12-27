import json
from functools import wraps
from typing import Any

import tomllib
import yaml

from utils.file.characteristics import get_file_size
from utils.file.exists import path_exists


class EmptyFileError(Exception):
    pass


def validate_file_exists(file_path: str) -> None:
    if not path_exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")


def validate_file_not_empty(file_path: str) -> None:
    if get_file_size(file_path) == 0:
        raise EmptyFileError(f"File {file_path} is empty")


def validate_file(f):
    VALIDATION_FNS = [validate_file_exists, validate_file_not_empty]

    @wraps(f)
    def inner(file_path: str, *args, **kwargs):
        nonlocal VALIDATION_FNS

        for validation_fn in VALIDATION_FNS:
            validation_fn(file_path)
        return f(file_path, *args, **kwargs)

    return inner


@validate_file
def read_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        data = f.read()
    return data


@validate_file
def load_json(file_path: str) -> Any:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


@validate_file
def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)
    return data


@validate_file
def load_toml(file_path: str) -> dict:
    with open(file_path, "rb") as f:
        data = tomllib.load(f)
    return data
