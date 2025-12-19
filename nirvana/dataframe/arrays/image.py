import sys
import base64
import requests # type: ignore
from io import BytesIO
from typing import Union, Sequence
import numpy as np
import pandas as pd
from PIL import Image
from pandas.api.extensions import ExtensionDtype, ExtensionArray


def load_image(raw_data: str | bytes | Image.Image | None) -> str | None:
    if raw_data is None:
        return None

    if isinstance(raw_data, Image.Image):
        buffered = BytesIO()
        raw_data.save(buffered, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    elif isinstance(raw_data, bytes):
        return "data:image/png;base64," + base64.b64encode(raw_data).decode("utf-8")
    elif isinstance(raw_data, str):
        if raw_data.startswith("data:image"):
            return raw_data
        elif raw_data.startswith("https://"):
            return raw_data
        elif raw_data.startswith("s3://"):
            from botocore.exceptions import NoCredentialsError, PartialCredentialsError

            try:
                import boto3

                s3 = boto3.client("s3")
                bucket_name, key = raw_data[5:].split("/", 1)  # Split after "s3://"
                response = s3.get_object(Bucket=bucket_name, Key=key)
                image_data = response["Body"].read()
                # is image_data bytes?
                image_obj = Image.open(BytesIO(image_data))
                buffered = BytesIO()
                image_obj.save(buffered, format="PNG")
                return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
            except (NoCredentialsError, PartialCredentialsError) as e:
                raise ValueError("AWS credentials not found or incomplete.") from e
            except Exception as e:
                raise ValueError(f"Failed to fetch image from S3: {e}") from e
        else:
            # it's a local file path
            with open(raw_data, "rb") as f:
                image_data = f.read()
                return "data:image/png;base64," + base64.b64encode(image_data).decode("utf-8")
    else:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64, S3, and PIL.Image, got {raw_data}")


class ImageDtype(ExtensionDtype):
    name = 'image'
    type = str
    na_value = None

    def __repr__(self):
        return "dtype('image')"

    @classmethod
    def construct_array_type(cls):
        return ImageArray


class ImageArray(ExtensionArray):
    def __init__(self, values):
        self._data = np.asarray(values, dtype=object)
        self._dtype = ImageDtype()

    def __getitem__(self, item: Union[int, slice, Sequence[int]]) -> np.ndarray:
        result = self._data[item]

        if isinstance(item, (int, np.integer)):
            # Return the raw value for display purposes
            return result

        return ImageArray(result)

    def __setitem__(self, key, value) -> None:
        """Set one or more values inplace, with cache invalidation."""
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                key = np.where(key)[0]
            key = key.tolist()
        if isinstance(key, (int, np.integer)):
            key = [key]
        if isinstance(key, slice):
            key = range(*key.indices(len(self)))
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            for idx, val in zip(key, value):
                self._data[idx] = load_image(val)
        else:
            for idx in key:
                self._data[idx] = load_image(value)

    def get_image(self, index: int) -> str | None:
        return self._data[index]

    def isna(self) -> np.ndarray:
        return pd.isna(self._data)

    def take(self, indices: Sequence[int], allow_fill: bool = False, fill_value=None) -> "ImageArray":
        result = self._data.take(indices, axis=0)
        if allow_fill and fill_value is not None:
            result[indices == -1] = fill_value
        return ImageArray(result)

    def copy(self) -> "ImageArray":
        new_array = ImageArray(self._data.copy())
        return new_array

    def _concat_same_type(cls, to_concat: Sequence["ImageArray"]) -> "ImageArray":
        """
        Concatenate multiple ImageArray instances into a single one.

        Args:
            to_concat (Sequence[ImageArray]): A sequence of ImageArray instances to concatenate.

        Returns:
            ImageArray: A new ImageArray containing all elements from the input arrays.
        """
        # create list of all data
        combined_data = np.concatenate([arr._data for arr in to_concat])
        return cls(combined_data)
    
    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if copy:
            scalars = np.array(scalars, dtype=object, copy=True)
        return cls(scalars)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:  # type: ignore
        if hasattr(other, "__iter__") and not isinstance(other, str):
            if len(other) != len(self):
                return np.repeat(False, len(self))
            return np.array([img1 == img2 for img1, img2 in zip(self._data, other)], dtype=bool)
        return np.array([img == other for img in self._data], dtype=bool)

    @property
    def dtype(self) -> ImageDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return sum(sys.getsizeof(val) for val in self._data if val)

    def __repr__(self) -> str:
        return f"ImageArray([{', '.join([f'<Image: {img}>' if img is not None else 'None' for img in self._data[:5]])}, ...])"

    def _formatter(self, boxed: bool = False):
        return lambda x: f"<Image: {x}>" if x is not None else "None"

    def to_numpy(self, dtype=None, copy=False, na_value=None) -> np.ndarray:
        """Convert the ImageArray to a numpy array."""
        return self._data

    def __array__(self, dtype=None) -> np.ndarray:
        """Numpy array interface."""
        return self.to_numpy(dtype=dtype)
