import os
import base64
import requests
from typing import Sequence
import numpy as np
from pandas.api.extensions import ExtensionDtype, ExtensionArray


def load_audio(raw_data: str | None) -> bytes | None:
    if raw_data is None:
        return None
    
    if raw_data.startswith("https://"):
        response = requests.get(raw_data)
        response.raise_for_status()
        audio_obj = response.content
        return base64.b64encode(audio_obj).decode('utf-8')
    elif os.path.isfile(raw_data):
        with open(raw_data, "rb") as f:
            file_obj = f.read()
            return base64.b64encode(file_obj).decode("utf-8")
    else:
        raise ValueError(f"Unrecognized file input, support local path, http url, and S3, got {raw_data}")


class AudioDtype(ExtensionDtype):
    """
    Dtype for AudioArray.
    """
    name = "audio"
    type = str
    na_value = None

    def __repr__(self):
        return "dtype('audio')"

    @classmethod
    def construct_array_type(cls):
        return AudioArray


class AudioArray(ExtensionArray):
    """
    An array where each element is a file path (a string).
    """
    def __init__(self, data):
        self._data = np.asarray(data, dtype=str)
        self._dtype = AudioDtype()

    def __getitem__(self, item: int | slice | Sequence[int]):
        if isinstance(item, int):
            return self._data[item]
        else:
            return AudioArray(self._data[item])

    def __len__(self):
        return len(self._data)

    @property
    def dtype(self) -> AudioDtype:
        return self._dtype

    def isna(self):
        return np.array([x is None for x in self._data], dtype=bool)

    def take(self, indices: Sequence[int], allow_fill=False, fill_value=None):
        data = self._data.take(self._data, indices, allow_fill=allow_fill, fill_value=fill_value)
        return AudioArray(data)

    def copy(self):
        return AudioArray(self._data.copy())

    def _concat_same_type(cls, to_concat):
        data = np.concatenate([arr._data for arr in to_concat])
        return cls(data)
