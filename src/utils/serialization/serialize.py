from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from typing import Any

import joblib


def serialize_object(object: Any) -> _TemporaryFileWrapper:
    with NamedTemporaryFile(delete=False) as fh:
        joblib.dump(object, fh)
    return fh
