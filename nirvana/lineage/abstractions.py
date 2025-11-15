import asyncio
from typing import Any, Union
from dataclasses import dataclass, field
import pandas as pd

from nirvana.ops.base import BaseOpOutputs
from nirvana.ops.scan import ScanOperation
from nirvana.ops.map import MapOperation, MapOpOutputs
from nirvana.ops.filter import FilterOperation, FilterOpOutputs
from nirvana.ops.rank import RankOperation, RankOpOutputs
from nirvana.ops.reduce import ReduceOperation, ReduceOpOutputs
from nirvana.ops.join import JoinOperation, JoinOpOutputs

OpOutputsType = Union[
    BaseOpOutputs, 
    MapOpOutputs, 
    FilterOpOutputs, 
    ReduceOpOutputs, 
    JoinOpOutputs,
    RankOpOutputs
]

op_mapping = {
    "scan": ScanOperation,
    "map": MapOperation,
    "filter": FilterOperation,
    "rank": RankOperation,
    "reduce": ReduceOperation,
    "join": JoinOperation,
}


@dataclass
class NodeFields:
    left_input_fields: list[str] = field(default_factory=list)
    right_input_fields: list[str] = field(default_factory=list)
    output_fields: list[str] = field(default_factory=list)


@dataclass
class DataSources:
    left_datasource: str = field(default="")
    right_datasource: str = field(default="")
    output_datasource: str = field(default="")


@dataclass
class NodeOutput:
    output: pd.DataFrame = field(default=None)
    cost: float = field(default=0.0)


class NodeBase:
    def __init__(self):
        self._left_parent: NodeBase | None = None
        self._right_parent: NodeBase | None = None

    @property
    def left_parent(self):
        return self._left_parent
    
    @property
    def right_parent(self):
        return self._right_parent
    
    def set_left_parent(self, node: "NodeBase"):
        self._left_parent = node

    def set_right_parent(self, node: "NodeBase"):
        self._right_parent = node


class LineageNode(NodeBase):
    def __init__(
            self, 
            op_name: str, 
            op_kwargs: dict,
            node_fields: dict,
            datasource: pd.DataFrame | None = None,
            **kwargs
    ):
        super().__init__()
        self.operator = op_mapping[op_name](**op_kwargs)
        self.node_fields = NodeFields(**node_fields)
        self.datasource = datasource

    @property
    def op_name(self) -> str:
        return self.operator.op_name

    async def execute_operation(self, input: pd.DataFrame | list[pd.DataFrame] | None) -> OpOutputsType:
        if self.op_name == "scan":
            return BaseOpOutputs(output=self.datasource, cost=0.0)
        elif self.op_name == "join":
            return await self.operator.execute(left_data=input[0], right_data=input[1])
        else:
            return await self.operator.execute(input_data=input)

    async def collate_dataframe(self, input: pd.DataFrame | list[pd.DataFrame] | None, op_outputs: OpOutputsType | None) -> pd.DataFrame:
        if self.op_name == "scan":
            return self.datasource
        elif self.op_name == "join":
            input[0]["keys"] = op_outputs.left_join_keys
            input[1]["keys"] = op_outputs.right_join_keys
            output = input[0].join(input[1], on="keys", how=self.operator.how).drop("keys", axis=1)
            return output
        elif self.op_name == "filter":
            if op_outputs.output is None:
                return input
            return input[op_outputs.output]
        elif self.op_name == "map":
            if op_outputs.output is None:
                return input
            input[op_outputs.field_name] = op_outputs.output
            return input
        elif self.op_name == "reduce":
            return pd.DataFrame({"reduce_result": [op_outputs.output]})

    async def run(self, input: pd.DataFrame | list[pd.DataFrame] | None) -> NodeOutput:
        if self.op_name == "scan":
            return NodeOutput(output=self.datasource, cost=0.0)
        
        elif self.op_name == "join":
            op_outputs = await self.operator.execute(left_data=input[0], right_data=input[1])
            input[0]["keys"] = op_outputs.left_join_keys
            input[1]["keys"] = op_outputs.right_join_keys
            output = input[0].join(input[1], on="keys", how=self.operator.how).drop("keys", axis=1)
            return NodeOutput(output=output, cost=op_outputs.cost)

        elif self.op_name == "filter":
            op_outputs = await self.operator.execute(input_data=input)
            if op_outputs.output is None:
                return NodeOutput(output=input, cost=op_outputs.cost)
            return NodeOutput(output=input[op_outputs.output], cost=op_outputs.cost)
        
        elif self.op_name == "map":
            op_outputs = await self.operator.execute(input_data=input)
            if op_outputs.output is None:
                return NodeOutput(output=input, cost=op_outputs.cost)
            input[op_outputs.field_name] = op_outputs.output
            return NodeOutput(output=input, cost=op_outputs.cost)
        
        elif self.op_name == "reduce":
            op_outputs = await self.operator.execute(input_data=input)
            return NodeOutput(output=pd.DataFrame({"reduce_result": [op_outputs.output]}), cost=op_outputs.cost)
        
        else:
            raise ValueError(f"Unsupported operator: {self.op_name}")
