"""
Reduce: aggregate multiple data based on NL predicates
"""
import mahjong as mjg
from mahjong.ops.base import BaseOperation


def reduce_helper(**kwargs):
    pass


class ReduceOperation(BaseOperation):
    """ TODO: Implement ReduceOperation class """

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("reduce", *args, **kwargs)
        pass

    def execute(self, *args, **kwargs):
        pass
