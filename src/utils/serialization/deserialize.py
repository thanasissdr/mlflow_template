from tempfile import _TemporaryFileWrapper
from typing import Any

import joblib


def deserialize_object(temp: _TemporaryFileWrapper) -> Any:
    with open(temp.name, "rb") as fh:
        fh.seek(0)
    return joblib.load(temp.name)
