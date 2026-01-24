import os
import base64
from typing import Sequence
import numpy as np
from pandas.api.extensions import ExtensionDtype, ExtensionArray


def load_file(raw_data: str | None) -> str | None:
    if raw_data is None:
        return None
    
    if raw_data.startswith("https://"):
        return raw_data
    elif raw_data.startswith("s3://"):
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError

        try:
            import boto3

            s3 = boto3.client("s3")
            bucket_name, key = raw_data[5:].split("/", 1)  # Split after "s3://"
            response = s3.get_object(Bucket=bucket_name, Key=key)
            file_obj = response["Body"].read()
            return f"data:application/pdf;base64,{file_obj}"
        except (NoCredentialsError, PartialCredentialsError):
            return None
    elif os.path.isfile(raw_data):
        with open(raw_data, "rb") as f:
            file_obj = f.read()
            return f"data:application/pdf;base64,{base64.b64encode(file_obj).decode("utf-8")}"
    else:
        raise ValueError(f"Unrecognized file input, support local path, http url, and S3, got {raw_data}")


class FileDtype(ExtensionDtype):
    """
    Dtype for FileArray.
    """
    name = "file"
    type = str
    na_value = None

    def __repr__(self):
        return "dtype('file')"

    @classmethod
    def construct_array_type(cls):
        return FileArray


class FileArray(ExtensionArray):
    """
    An array where each element is a file path (a string).
    """
    def __init__(self, data):
        self._data = np.asarray(data, dtype=str)
        self._dtype = FileDtype()

    def __getitem__(self, item: int | slice | Sequence[int]):
        if isinstance(item, int):
            return self._data[item]
        else:
            return FileArray(self._data[item])

    def __len__(self):
        return len(self._data)

    @property
    def dtype(self) -> FileDtype:
        return self._dtype

    def isna(self):
        return np.array([x is None for x in self._data], dtype=bool)

    def take(self, indices: Sequence[int], allow_fill=False, fill_value=None):
        data = self._data.take(self._data, indices, allow_fill=allow_fill, fill_value=fill_value)
        return FileArray(data)

    def copy(self):
        return FileArray(self._data.copy())

    def _concat_same_type(cls, to_concat):
        data = np.concatenate([arr._data for arr in to_concat])
        return cls(data)
