from typing import Sequence
import numpy as np
from pandas.api.extensions import ExtensionDtype, ExtensionArray


class FileDtype(ExtensionDtype):
    """
    Dtype for FileArray.
    """
    name = "file"
    type = str
    na_value = None

    @classmethod
    def construct_array_type(cls):
        return FileArray


class FileArray(ExtensionArray):
    """
    An array where each element is a file path (a string).
    """
    def __init__(self, data):
        self._data = np.asarray(data, dtype=str)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._data[item]
        return FileArray(self._data[item])

    def __len__(self):
        return len(self._data)

    def __eq__(self, other):
        if isinstance(other, FileArray):
            return np.array_equal(self._data, other._data)
        return False

    @property
    def dtype(self):
        return FileDtype()

    def isna(self):
        return np.array([x is None for x in self._data], dtype=bool)

    def take(self, indices: Sequence[int], allow_fill=False, fill_value=None):
        data = self._data.take(self._data, indices, allow_fill=allow_fill, fill_value=fill_value)
        return FileArray(data)

    def copy(self):
        return FileArray(self._data.copy())

    @classmethod
    def _concat_same_type(cls, to_concat):
        data = np.concatenate([arr._data for arr in to_concat])
        return cls(data)
