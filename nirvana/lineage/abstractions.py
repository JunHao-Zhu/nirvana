import asyncio
from typing import List, Union, Callable, Any

import pandas as pd
from pydantic import BaseModel

from nirvana.ops.map import MapOperation, MapOpOutputs
from nirvana.ops.filter import FilterOperation, FilterOpOutputs
from nirvana.ops.reduce import ReduceOperation, ReduceOpOutputs
from nirvana.ops.join import JoinOperation, JoinOpOutputs

OpOutputsType = Union[MapOpOutputs, FilterOpOutputs, ReduceOpOutputs, JoinOpOutputs]

op_mapping = {
    "map": MapOperation,
    "filter": FilterOperation,
    "reduce": ReduceOperation,
    "join": JoinOperation
}


class NodeOutput(BaseModel):
    output: pd.DataFrame | List[Any]
    cost: float
    

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
    
    def set_left_parent(self, node: "NodeBase" | None):
        self._left_parent = node

    def set_right_parent(self, node: "NodeBase" | None):
        self._right_parent = node


class LineageNode(NodeBase):
    def __init__(
            self, 
            op_name: str | None = None, 
            op_metadata: dict | None = None,
            data_metadata: dict | None = None,
            datasource: pd.DataFrame | None = None,
            func: Callable = None,
            **kwargs
    ):
        super().__init__()
        self.op_name = op_name
        self.op_metadata = op_metadata
        self.data_metadata = data_metadata

        self.op = op_mapping[op_name](
            **kwargs
        )
        self.func = func
        self.exec_model = None
        self.datasource = datasource
        
    def set_exec_model(self, model_name: str):
        self.exec_model = model_name

    async def execute_operation(self, input: pd.DataFrame | List[pd.DataFrame] | None) -> NodeOutput:
        if self.func is not None:
            self.op_kwargs["func"] = self.func
        if self.exec_model is not None:
            self.op_kwargs["model"] = self.exec_model
        
        if self.op_name == "scan":
            return NodeOutput(output=self.datasource, cost=0.0)
        elif self.op_name == "join":
            return await self.op.execute(left_data=input[0], right_data=input[1], **self.op_kwargs)
        else:
            return await self.op.execute(input_data=input, **self.op_kwargs)
        
    async def collate_dataframe(self, input: pd.DataFrame | List[pd.DataFrame] | None, op_outputs: NodeOutput | None) -> pd.DataFrame:
        if self.op_name == "scan":
            return op_outputs
        elif self.op_name == "join":
            input[0]["keys"] = op_outputs.left_join_keys
            input[1]["keys"] = op_outputs.right_join_keys
            output = input[0].join(input[1], on="keys", how=self.op_kwargs["how"]).drop("keys", axis=1)
            return NodeOutput(output=output, cost=op_outputs.cost)
        elif self.op_name == "filter":
            if op_outputs.output is None:
                return NodeOutput(output=input, cost=op_outputs.cost)
            return NodeOutput(output=input[op_outputs.output], cost=op_outputs.cost)
        elif self.op_name == "map":
            if op_outputs.output is None:
                return NodeOutput(output=input, cost=op_outputs.cost)
            input[op_outputs.new_field] = op_outputs.output
            return NodeOutput(output=input, cost=op_outputs.cost)
        elif self.op_name == "reduce":
            return NodeOutput(output=pd.DataFrame({"reduce_result": [op_outputs.output]}), cost=op_outputs.cost)

    async def run(self, input: pd.DataFrame | List[pd.DataFrame] | None) -> NodeOutput:
        if self.func is not None:
            self.op_kwargs["func"] = self.func
        if self.exec_model is not None:
            self.op_kwargs["model"] = self.exec_model

        if self.op_name == "scan":
            return NodeOutput(output=self.datasource, cost=0.0)
        
        elif self.op_name == "join":
            op_outputs = await self.op.execute(left_data=input[0], right_data=input[1], **self.op_kwargs)
            input[0]["keys"] = op_outputs.left_join_keys
            input[1]["keys"] = op_outputs.right_join_keys
            output = input[0].join(input[1], on="keys", how=self.op_kwargs["how"]).drop("keys", axis=1)
            return NodeOutput(output=output, cost=op_outputs.cost)

        elif self.op_name == "filter":
            op_outputs = await self.op.execute(input_data=input, **self.op_kwargs)
            if op_outputs.output is None:
                return NodeOutput(output=input, cost=op_outputs.cost)
            return NodeOutput(output=input[op_outputs.output], cost=op_outputs.cost)
        
        elif self.op_name == "map":
            op_outputs = await self.op.execute(input_data=input, **self.op_kwargs)
            if op_outputs.output is None:
                return NodeOutput(output=input, cost=op_outputs.cost)
            input[op_outputs.new_field] = op_outputs.output
            return NodeOutput(output=input, cost=op_outputs.cost)
        
        elif self.op_name == "reduce":
            op_outputs = await self.op.execute(input_data=input, **self.op_kwargs)
            return NodeOutput(output=pd.DataFrame({"reduce_result": [op_outputs.output]}), cost=op_outputs.cost)
        
        else:
            raise ValueError(f"Unspported operator: {self.op_name}")
