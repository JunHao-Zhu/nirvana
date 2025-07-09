import asyncio
from typing import Any, List, Iterable, Tuple
from dataclasses import dataclass

from nirvana.ops.base import BaseOpOutputs, BaseOperation


def discover_wrapper(**kwargs):
    pass


class DiscoverOperation(BaseOperation):
    """ TODO: Implement data discover 
    Operation for data discovery that identify relevant data(tables) from data lakes for a data analytics task (in NL).
    
    The algorithm is adopted from [Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System](https://arxiv.org/abs/2504.09207)
    """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("discover", *args, **kwargs)
