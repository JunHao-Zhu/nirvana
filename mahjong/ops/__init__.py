import importlib.metadata

from mahjong.ops.discover import DiscoverOperation
from mahjong.ops.filter import FilterOperation
from mahjong.ops.map import MapOperation
from mahjong.ops.rank import RankOperation
from mahjong.ops.reduce import ReduceOperation
from mahjong.ops.select import SelectOperation


mapping = {
    "discover": DiscoverOperation,
    "filter": FilterOperation,
    "map": MapOperation,
    "rank": RankOperation,
    "reduce": ReduceOperation,
    "select": SelectOperation,
}

def get_operation(operation_type: str):
    """Loads a single operation by name""" 
    try:
        entrypoint = importlib.metadata.entry_points(group="docetl.operation")[
            operation_type
        ]
        return entrypoint.load()
    except KeyError:
        if operation_type in mapping:
            return mapping[operation_type]
        raise KeyError(f"Unrecognized operation {operation_type}")

def get_operations():
    """Load all available operations and return them as a dictionary"""
    operations = mapping.copy()
    operations.update({
        op.name: op.load()
        for op in importlib.metadata.entry_points(group="docetl.operation")
    })
    return operations
