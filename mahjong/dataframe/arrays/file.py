import re
import base64
from typing import Sequence, Union

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
        self._data = self._convert_base64_data(data)

    def _convert_to_base64(self, path_list: Union[Sequence[str], str]) -> np.ndarray:
        if isinstance(path_list, str):
            path_list = [path_list]
        
        base64_pattern = re.compile(r"^data:application/pdf;base64,[a-zA-Z0-9+/=]+$")
        
        def _validate_and_convert(path):
            if base64_pattern.match(path):
                return path  # Already in base64 format
            try:
                with open(path, "rb") as pdf_file:
                    encoded_string = base64.b64encode(pdf_file.read()).decode("utf-8")
                    return f"data:application/pdf;base64,{encoded_string}"
            except Exception as e:
                raise ValueError(f"Error converting file {path} to base64: {e}")
        
        return np.array([_validate_and_convert(path) for path in path_list], dtype=str)

    def __getitem__(self, item: Union[int, slice, Sequence[int]]) -> np.ndarray:
            result = self._data[item]

            if isinstance(item, (int, np.integer)):
                # Return the raw value for display purposes
                return result

            return FileArray(result)

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
