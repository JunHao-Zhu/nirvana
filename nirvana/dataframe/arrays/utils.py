import os
import multiprocessing
from multiprocessing import Pool
import pandas as pd

from nirvana.dataframe.arrays.image import ImageDtype, ImageArray, load_image
from nirvana.dataframe.arrays.audio import AudioDtype, AudioArray, load_audio
from nirvana.dataframe.arrays.file import FileDtype, FileArray, load_file


EXT_TYPE_MAP = {
    "image": [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp", ".svg", ".ico", ".jfif"],
    "audio": [".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".opus"],
    "video": [".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"],
    "pdf": [".pdf"],
    "doc": [".doc", ".docx", ".txt", ".rtf", ".odt"],
    "ppt": [".ppt", ".pptx", ".odp"],
    # "excel": [".xls", ".xlsx", ".csv"],
}


def infer_and_convert_dtype(col: pd.Series) -> tuple[pd.Series, type]:
    """
    Infer and return a specific Python or extension dtype/type (e.g., int, float, str, ImageArray, AudioArray, etc.)
    for a given column-like structure. Distinguish between pandas/pyarrow types and unstructured
    data, such as image/audio file URLs and paths.
    """
    if col.empty:
        return col, col.dtype
    
    dtype = col.dtype
    if pd.api.types.is_string_dtype(dtype):
        _, ext = os.path.splitext(col.iloc[0])
        if ext in EXT_TYPE_MAP["image"]:
            num_workers = multiprocessing.cpu_count()
            with Pool(num_workers) as pool:
                col = pool.map(load_image, col.values)
            return ImageArray(col), ImageDtype
        elif ext in EXT_TYPE_MAP["audio"]:
            num_workers = multiprocessing.cpu_count()
            with Pool(num_workers) as pool:
                col = pool.map(load_audio, col.values)
            return AudioArray(col), AudioDtype
        elif ext in EXT_TYPE_MAP["pdf"]:
            num_workers = multiprocessing.cpu_count()
            with Pool(num_workers) as pool:
                col = pool.map(load_file, col.values)
            return FileArray(col), FileDtype
        else:
            return col, str
    else:
        return col, dtype
