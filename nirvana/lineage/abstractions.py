import asyncio
import time
from typing import Union
from dataclasses import dataclass, field
import pandas as pd

from nirvana.ops.base import BaseOperation, BaseOpOutputs
from nirvana.ops.scan import ScanOperation, ScanOpOutputs
from nirvana.ops.map import MapOperation, MapOpOutputs
from nirvana.ops.filter import FilterOperation, FilterOpOutputs
from nirvana.ops.rank import RankOperation, RankOpOutputs
from nirvana.ops.reduce import ReduceOperation, ReduceOpOutputs
from nirvana.ops.join import JoinOperation, JoinOpOutputs

OpOutputsType = Union[
    BaseOpOutputs,
    ScanOpOutputs,
    MapOpOutputs, 
    FilterOpOutputs, 
    ReduceOpOutputs, 
    JoinOpOutputs,
    RankOpOutputs
]

OpMapping = {
    "scan": ScanOperation,
    "map": MapOperation,
    "filter": FilterOperation,
    "rank": RankOperation,
    "reduce": ReduceOperation,
    "join": JoinOperation,
}


schema_mapping = {
    "scan": "SCAN({source}):[]->[{output_columns}]",
    "map": "MAP({user_instruction}):[{input_columns}]->[{output_columns}]",
    "filter": "FILTER({user_instruction}):[{input_columns}]->[Bool]",
    "rank": "RANK({user_instruction}):[{input_column}]->[Rank]",
    "reduce": "AGGR({user_instruction}):[{input_column}]->[Aggr]",
    "join": "{how}-JOIN({user_instruction}):[left[{left_on}] * right[{right_on}]",
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
        self._left_child: NodeBase | None = None
        self._right_child: NodeBase | None = None

    @property
    def left_child(self):
        return self._left_child
    
    @property
    def right_child(self):
        return self._right_child
    
    def set_left_child(self, node: "NodeBase"):
        self._left_child = node

    def set_right_child(self, node: "NodeBase"):
        self._right_child = node


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
        self.operator: BaseOperation = OpMapping[op_name](**op_kwargs)
        self.node_fields = NodeFields(**node_fields)
        self.datasource = datasource

    @property
    def op_name(self) -> str:
        return self.operator.op_name

    async def execute_operation(self, input: pd.DataFrame | list[pd.DataFrame] | None = None) -> OpOutputsType:
        if self.op_name == "scan":
            return await self.operator.execute(input_data=self.datasource)
        elif self.op_name == "join":
            return await self.operator.execute(left_data=input[0], right_data=input[1])
        else:
            return await self.operator.execute(input_data=input)

    async def collate_dataframe(self, input: pd.DataFrame | list[pd.DataFrame] | None = None, op_outputs: OpOutputsType | None = None) -> pd.DataFrame:
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
            if op_outputs.outputs:
                output = pd.concat([input, pd.DataFrame(op_outputs.outputs)], axis=1)
                return output
            else:
                return input
        elif self.op_name == "rank":
            return input.iloc[op_outputs.ranked_indices]
        elif self.op_name == "reduce":
            return pd.DataFrame({"reduce_result": [op_outputs.output]})

    async def run(self, input: pd.DataFrame | list[pd.DataFrame] | None = None) -> NodeOutput:
        if self.op_name == "scan":
            op_outputs = await self.operator.execute(input_data=self.datasource)
            return NodeOutput(output=op_outputs.output, cost=op_outputs.cost)
        
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
            if op_outputs.outputs:
                output = pd.concat([input, pd.DataFrame(op_outputs.outputs)], axis=1)
                return NodeOutput(output=output, cost=op_outputs.cost)
            else:
                return NodeOutput(output=input, cost=op_outputs.cost)
        
        elif self.op_name == "rank":
            op_outputs = await self.operator.execute(input_data=input)
            output = input.iloc[op_outputs.ranked_indices]
            return NodeOutput(output=output, cost=op_outputs.cost)
        
        elif self.op_name == "reduce":
            op_outputs = await self.operator.execute(input_data=input)
            return NodeOutput(output=pd.DataFrame({"reduce_result": [op_outputs.output]}), cost=op_outputs.cost)
        
        else:
            raise ValueError(f"Unsupported operator: {self.op_name}")


def collect_op_metadata(op_node: LineageNode, max_instruction_print_length: int = 256):
    op_name = op_node.op_name
    op_kwargs = op_node.operator.op_kwargs
    user_instruction = op_kwargs["user_instruction"][:max_instruction_print_length]
    if op_name == "map":
        dependencies = op_kwargs["input_columns"][:5]
        dependencies_str = ", ".join(dependencies) + ", ..." if len(dependencies) > 5 else ", ".join(dependencies)
        generated_fields = op_kwargs["output_columns"][:5]
        generated_fields_str = ", ".join(generated_fields) + ", ..." if len(generated_fields) > 5 else ", ".join(generated_fields)
        return (
            schema_mapping[op_name].format(user_instruction=user_instruction, input_columns=dependencies_str, output_columns=generated_fields_str)
        )
    elif op_name == "filter":
        dependencies = op_kwargs["input_columns"][:5]
        dependencies_str = ", ".join(dependencies) + ", ..." if len(dependencies) > 5 else ", ".join(dependencies)
        return (
            schema_mapping[op_name].format(user_instruction=user_instruction, input_columns=dependencies_str)
        )
    elif op_name in ["rank", "reduce"]:
        return (
            schema_mapping[op_name].format(user_instruction=user_instruction, input_column=op_kwargs['input_columns'][0])
        )
    elif op_name == "join":
        return (
            schema_mapping[op_name].format(how=op_kwargs['how'], user_instruction=user_instruction, left_on=op_kwargs['left_on'][0], right_on=op_kwargs['right_on'][0])
        )
    elif op_name == "scan":
        output_columns = op_kwargs["output_columns"][:2]
        output_columns_str = ", ".join(output_columns) + ", ..."
        return (
            schema_mapping[op_name].format(source=op_kwargs['source'], output_columns=output_columns_str)
        )
    return ""


def execute_along_lineage(leaf_node: LineageNode):
    def _execute_node(node: LineageNode, token_cost: float) -> tuple[pd.DataFrame, float]:
        if node.left_child:
            left_node_output, cost_from_left_subtree = _execute_node(node.left_child, token_cost)
        if node.right_child:
            right_node_output, cost_from_right_subtree = _execute_node(node.right_child, token_cost)
        
        if node.op_name == "scan":
            node_output = asyncio.run(node.run())
            accumulated_cost = node_output.cost
            return node_output.output, accumulated_cost
        
        elif node.op_name == "join":
            node_output = asyncio.run(node.run([left_node_output, right_node_output]))
            accumulated_cost = cost_from_left_subtree + cost_from_right_subtree + node_output.cost
            return node_output.output, accumulated_cost
        
        else:
            node_output = asyncio.run(node.run(left_node_output))
            accumulated_cost = cost_from_left_subtree + node_output.cost
            return node_output.output, accumulated_cost
    
    execution_start_time = time.time()
    output_from_lineage, total_token_cost = _execute_node(leaf_node, 0.0)
    execution_end_time = time.time()
    execution_time = execution_end_time - execution_start_time
    return output_from_lineage, total_token_cost, execution_time
