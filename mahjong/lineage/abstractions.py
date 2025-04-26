from typing import List, Union
import pandas as pd

from mahjong.ops.map import MapOperation, MapOpOutputs
from mahjong.ops.filter import FilterOperation, FilterOpOutputs
from mahjong.ops.reduce import ReduceOperation, ReduceOpOutputs

OpOutputsType = Union[MapOpOutputs, FilterOpOutputs, ReduceOpOutputs]

op_mapping = {
    "map": MapOperation,
    "filter": FilterOperation,
    "reduce": ReduceOperation
}


class LineageNode:
    def __init__(self):
        self.is_visited = False
        self._parent: List[LineageNode] = []
        self._child: List[LineageNode] = []

    @property
    def parent(self):
        return self._parent
    
    @property
    def child(self):
        return self._child
    
    def set_parent(self, nodes: List["LineageNode"]):
        self._parent = nodes.copy()

    def set_child(self, nodes: List["LineageNode"]):
        self._child = nodes.copy()

    def remove_child(self, node: "LineageNode"):
        self._child.remove(node)

    def remove_parent(self, node: "LineageNode"):
        self._parent.remove(node)
    
    def add_child(self, node: "LineageNode"):
        self._child.append(node)

    def add_parent(self, node: "LineageNode"):
        self._parent.append(node)
    

class LineageDataNode(LineageNode):
    def __init__(
            self, 
            columns: List[str], 
            new_field: Union[str, None] = None, 
            materialized: bool = False
    ):
        super().__init__()
        self.columns = columns
        self.new_field = new_field
        self.materialized = materialized

    def run(self, input_data: pd.DataFrame, last_op_output: OpOutputsType) -> pd.DataFrame:
        if hasattr(last_op_output, "field_name"):
            self.new_field = last_op_output.field_name
            if last_op_output.output is None:
                return input_data
            input_data[self.new_field] = last_op_output.output
            return input_data
        if isinstance(last_op_output, FilterOpOutputs):
            if last_op_output.output is None:
                return input_data
            input_data = input_data[last_op_output.output]
            return input_data
        if isinstance(last_op_output, ReduceOpOutputs):
            return pd.DataFrame({"reduce_result": [last_op_output.output]})


class LineageOpNode(LineageNode):
    def __init__(
            self, 
            op_name: str, 
            user_instruction: Union[str, List[str]],
            input_column: str,
            output_column: str = None
    ):
        super().__init__()
        self.op_name = op_name
        self.op = op_mapping[op_name]()
        self.user_instruction = user_instruction
        self.input_column = input_column
        self.output_column = output_column

    def run(self, input_data: pd.DataFrame) -> OpOutputsType:
        return self.op.execute(
            input_data=input_data,
            input_column=self.input_column,
            user_instruction=self.user_instruction,
            output_column=self.output_column
        )
