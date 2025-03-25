"""
Filter: remove data that violates NL predicates.
"""
import mahjong as mj
from ops.base import BaseOperation


class FilterOperation(BaseOperation):
    """ TODO: Implement FilterOperation class """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("filter", *args, **kwargs)
        pass
