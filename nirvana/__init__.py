import logging
from typing import Union
import pandas as pd

from nirvana.dataframe.frame import DataFrame
from nirvana.dataframe.arrays.image import ImageArray, ImageDtype
from nirvana.utils import configure_llm_backbone

pd.api.extensions.register_extension_dtype(ImageDtype)


logging.basicConfig(format="[\033[34m%(asctime)s\033[0m] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_base_data(data: Union[pd.Series, list]) -> list:
    """
    Converts data to proper base data type.
    - For original pandas data types, this is returns tolist().
    - For ImageDtype, this returns list of PIL.Image.Image.
    """
    if isinstance(data, pd.Series):
        if isinstance(data.dtype, ImageDtype):
            return [data.array.get_image(i) for i in range(len(data))]
        return data.tolist()

    return data

__all__ = [
    "logger",
    "DataFrame",
    "ImageDtype", 
    "ImageArray", 
    "convert_to_base_data", 
    "configure_llm_backbone"
]
