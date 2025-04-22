from mahjong.ops.discover import discover_wrapper as discover
from mahjong.ops.filter import filter_wrapper as filter
from mahjong.ops.map import map_wrapper as map
from mahjong.ops.rank import rank_wrapper as rank
from mahjong.ops.reduce import reduce_wrapper as reduce
from mahjong.ops.select import select_wrapper as select
from mahjong.ops.join import join_wrapper as join


__all__ = [
    "discover",
    "filter",
    "map",
    "rank",
    "reduce",
    "select",
    "join"
]
