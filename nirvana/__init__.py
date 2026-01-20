import sys
import logging
from typing import Union
import pandas as pd

from nirvana.dataframe.frame import DataFrame
from nirvana.dataframe.arrays.image import ImageArray, ImageDtype
from nirvana.dataframe.arrays.audio import AudioArray, AudioDtype
from nirvana.dataframe.arrays.file import FileArray, FileDtype
from nirvana.utils import configure_llm_backbone

pd.api.extensions.register_extension_dtype(ImageDtype)
pd.api.extensions.register_extension_dtype(AudioDtype)
pd.api.extensions.register_extension_dtype(FileDtype)

__name__ = "nirvana"


class NirvanaLoggingStream:
    """
    A custom logging APIs throughout Nirvana (e.g., logger.info, etc.). This stream is a wrapper 
    of stderr. It also provides capability to disable the stream to silence the event logs.
    """

    def __init__(self):
        self._enabled = True

    def write(self, message):
        if self._enabled:
            sys.stderr.write(message)

    def flush(self):
        if self._enabled:
            sys.stderr.flush()

    @property
    def enabled(self):
        return self._enabled
    
    @enabled.setter
    def enabled(self, value):
        self._enabled = value


def disable_logging():
    """ Disable the logging stream to silence the event logs. """
    NirvanaLoggingStream.enabled = False


def enable_logging():
    """ Enable the logging stream to enable the event logs. """
    NirvanaLoggingStream.enabled = True


def configure_nirvana_loggers(root_module_name):
    formatter = logging.Formatter(
        fmt="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    nirvana_handler_name = "nirvana_handler"
    handler = logging.StreamHandler(stream=NirvanaLoggingStream())
    handler.setFormatter(formatter)
    handler.set_name(nirvana_handler_name)
    
    logger = logging.getLogger(root_module_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    for existing_handler in logger.handlers[:]:
        if getattr(existing_handler, "name", None) == nirvana_handler_name:
            logger.removeHandler(existing_handler)
    
    logger.addHandler(handler)


configure_nirvana_loggers(__name__)


__all__ = [
    "logger",
    "DataFrame",
    "ImageDtype", 
    "ImageArray", 
    "AudioDtype", 
    "AudioArray", 
    "FileDtype", 
    "FileArray",
    "configure_llm_backbone"
]
