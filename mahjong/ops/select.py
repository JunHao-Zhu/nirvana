"""
Select: Extract relevant information from data based on NL predicates.
"""
import mahjong as mjg
from mahjong.ops.base import BaseOperation


def select_wrapper(**kwargs):
    pass


class SelectOperation(BaseOperation):
    """ TODO: Implement SelectOperation class """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("select", *args, **kwargs)
        pass
