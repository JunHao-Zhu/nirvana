"""
Reduce: aggregate multiple data based on NL predicates
"""
import mahjong as mj
from ops.base import BaseOperation


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
