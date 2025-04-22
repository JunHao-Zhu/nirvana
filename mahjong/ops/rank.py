"""
Rank: Rank data according to NL predicates.
"""

from mahjong.ops.base import BaseOperation


def rank_wrapper(**kwargs):
    pass


class RankOperation(BaseOperation):
    """ TODO: Implement RankOperation class """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("rank", *args, **kwargs)
        pass
