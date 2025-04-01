"""
DataFrame class for handling data in a tabular format.
To begin, DataFrame is a combination of regular pandas DataFrame and additional unstructured data types (e.g., text, images, and audio).
Example:
| location | house price | comment | house picture |
|----------|-------------|---------|---------------|
|161 Auburn St. Unit 161, Cambridge, MA 02139| $1,000,000 | "Great location!" | ![house](house.jpg) (a link to the house image) |
|...|...|...|...|

A simple version that directly uses pandas DataFrame and supports image data type through pandas api extension, like lotus.
Later, we will implement an independent DataFrame of pandas DataFrame. The usage will be like:
```python
import mahjong as mjg

df = mjg.DataFrame({
    'location': ['161 Auburn St. Unit 161, Cambridge, MA 02139'],
    'house price': ['$1,000,000'],
    'comment': ['Great location!'],
    'house picture': [mjg.Image('house.jpg')]
})
```
Moreover, in the future, we consider reading data from data lake storages (e.g, S3, Delta Lake, etc.)
"""

import sys
import base64
import requests # type: ignore
from io import BytesIO
from typing import Union, Sequence

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype, ExtensionArray
from PIL import Image

from mahjong.lineage.lineage import LineageMixin


def fetch_image(image: Union[str, np.ndarray, Image.Image, None], image_type: str = "Image") -> Union[Image.Image, str, None]:
    if image is None:
        return None

    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, np.ndarray):
        image_obj = Image.fromarray(image.astype("uint8"))
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    elif image.startswith("s3://"):
        from botocore.exceptions import NoCredentialsError, PartialCredentialsError

        try:
            import boto3

            s3 = boto3.client("s3")
            bucket_name, key = image[5:].split("/", 1)  # Split after "s3://"
            response = s3.get_object(Bucket=bucket_name, Key=key)
            image_data = response["Body"].read()
            image_obj = Image.open(BytesIO(image_data))
        except (NoCredentialsError, PartialCredentialsError) as e:
            raise ValueError("AWS credentials not found or incomplete.") from e
        except Exception as e:
            raise ValueError(f"Failed to fetch image from S3: {e}") from e
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(
            f"Unrecognized image input, support local path, http url, base64, S3, and PIL.Image, got {image}"
        )
    image_obj = image_obj.convert("RGB")
    if image_type == "base64":
        buffered = BytesIO()
        image_obj.save(buffered, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    return image_obj


class ImageDtype(ExtensionDtype):
    name = 'image'
    type = Image.Image
    na_value = None

    @classmethod
    def construct_array_type(cls):
        return ImageArray
    

class ImageArray(ExtensionArray):
    def __init__(self, values):
        self._data = np.asarray(values, dtype=object)
        self._dtype = ImageDtype()
        self.allowed_image_types = ["Image", "base64"]
        self._cached_images: dict[tuple[int, str], Union[str, Image.Image, None]] = {}  # Cache for loaded images

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
                self._data[idx] = val
                self._invalidate_cache(idx)
        else:
            for idx in key:
                self._data[idx] = value
                self._invalidate_cache(idx)

    def _invalidate_cache(self, idx: int) -> None:
        """Remove an item from the cache."""
        for image_type in self.allowed_image_types:
            if (idx, image_type) in self._cached_images:
                del self._cached_images[(idx, image_type)]

    def get_image(self, idx: int, image_type: str = "Image") -> Union[Image.Image, str, None]:
        """Explicit method to fetch and return the actual image"""
        if (idx, image_type) not in self._cached_images:
            image_result = fetch_image(self._data[idx], image_type)
            assert image_result is None or isinstance(image_result, (Image.Image, str))
            self._cached_images[(idx, image_type)] = image_result
        return self._cached_images[(idx, image_type)]

    def isna(self) -> np.ndarray:
        return pd.isna(self._data)

    def take(self, indices: Sequence[int], allow_fill: bool = False, fill_value=None) -> "ImageArray":
        result = self._data.take(indices, axis=0)
        if allow_fill and fill_value is not None:
            result[indices == -1] = fill_value
        return ImageArray(result)

    def copy(self) -> "ImageArray":
        new_array = ImageArray(self._data.copy())
        new_array._cached_images = self._cached_images.copy()
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
        return cls._from_sequence(combined_data)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        if copy:
            scalars = np.array(scalars, dtype=object, copy=True)
        return cls(scalars)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other) -> np.ndarray:  # type: ignore
        if isinstance(other, ImageArray):
            return np.array([_compare_images(img1, img2) for img1, img2 in zip(self._data, other._data)], dtype=bool)

        if hasattr(other, "__iter__") and not isinstance(other, str):
            if len(other) != len(self):
                return np.repeat(False, len(self))
            return np.array([_compare_images(img1, img2) for img1, img2 in zip(self._data, other)], dtype=bool)
        return np.array([_compare_images(img, other) for img in self._data], dtype=bool)

    @property
    def dtype(self) -> ImageDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return sum(sys.getsizeof(img) for img in self._data if img)

    def __repr__(self) -> str:
        return f"ImageArray([{', '.join([f'<Image: {type(img)}>' if img is not None else 'None' for img in self._data[:5]])}, ...])"

    def _formatter(self, boxed: bool = False):
        return lambda x: f"<Image: {type(x)}>" if x is not None else "None"

    def to_numpy(self, dtype=None, copy=False, na_value=None) -> np.ndarray:
        """Convert the ImageArray to a numpy array."""
        pil_images = []
        for i, img_data in enumerate(self._data):
            if isinstance(img_data, np.ndarray):
                image = self.get_image(i)
                pil_images.append(image)
            else:
                pil_images.append(img_data)
        result = np.empty(len(self), dtype=object)
        result[:] = pil_images
        return result

    def __array__(self, dtype=None) -> np.ndarray:
        """Numpy array interface."""
        return self.to_numpy(dtype=dtype)


def _compare_images(img1, img2) -> bool:
    if img1 is None or img2 is None:
        return img1 is img2

    # Only fetch images when actually comparing
    if isinstance(img1, Image.Image) or isinstance(img2, Image.Image):
        img1 = fetch_image(img1)
        img2 = fetch_image(img2)
        return img1.size == img2.size and img1.mode == img2.mode and img1.tobytes() == img2.tobytes()
    else:
        return img1 == img2
    

@pd.api.extensions.register_dataframe_accessor("tile")
class Tile(LineageMixin):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def semantic_map(self, user_instruction):
        self.add_operator(op_name="map",
                          user_instruction=user_instruction)
        
    def semantic_filter(self, user_instruction):
        self.add_operator(op_name="filter",
                          user_instruction=user_instruction)
        
    def semantic_reduce(self, user_instruction):
        self.add_operator(op_name="reduce",
                          user_instruction=user_instruction)
        
    def execute(self):
        self.optimize()
        pass


class DataFrame:
    def __init__(
            self,
            data: Union[dict, list] = None,
            primary_key: Union[str, bool] = None,
            *args,
            **kwargs
    ):
        self.primary_key = None if primary_key is False else primary_key
        self.data = data

    def __len__(self):
        _len = self.nrows
        return _len
    
    def __contains__(self, item):
        return self.columns.__contains__(item)
    
    @property
    def columns(self):
        return list(self.data.keys())
    
    @property
    def primary_key(self):
        return self._primary_key
    
    @property
    def nrows(self):
        return len(self.data[self.primary_key])
        