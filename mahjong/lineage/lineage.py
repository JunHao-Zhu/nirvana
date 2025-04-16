"""
Record the OP lineage (operator and its user instruction) for optimizing operator orchestration.
"""

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
            input_data[self.new_field] = last_op_output.output
            return input_data
        if isinstance(last_op_output, FilterOpOutputs):
            input_data = input_data[last_op_output.output]
            return input_data
        if isinstance(last_op_output, ReduceOpOutputs):
            return pd.DataFrame({"reduce_result": last_op_output.output})


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


class LineageMixin:
    def __init__(self):
        self.last_node = None

    def add_operator(self, op_name, user_instruction, input_column, output_column=None, fields=None):
        op_node = LineageOpNode(
            op_name, user_instruction, input_column, output_column
        )
        if self.last_node is None:
            data_node = LineageDataNode(columns=fields, new_field=output_column)
            op_node.add_child(data_node)
            data_node.add_parent(op_node)
            self.last_node = data_node
        else:
            columns_from_last_node = (
                self.last_node.columns 
                if self.last_node.new_field is None else 
                self.last_node.columns + [self.last_node.new_field]
            )
            data_node = LineageDataNode(columns=columns_from_last_node, new_field=output_column)
            op_node.add_child(data_node)
            data_node.add_parent(op_node)

            op_node.add_parent(self.last_node)
            self.last_node.add_child(op_node)
            self.last_node = data_node

    def optimize(self):
        pass

    def execute(self, input_data: pd.DataFrame):
        dataframe_from_last_node = input_data.copy()
        output_from_last_node = None
        def _run_node(node: LineageNode):
            # if the node is the first operator, run it on input data
            if len(node.parent) == 0:
                assert isinstance(node, LineageOpNode), "The first node should be an operator."
                output_from_last_node = node.run(dataframe_from_last_node)
                return
            # run the parent nodes
            for parent_node in node.parent:
                _run_node(parent_node)

            if isinstance(node, LineageDataNode):
                dataframe_from_last_node = node.run(dataframe_from_last_node, output_from_last_node)
                return

            if isinstance(node, LineageOpNode):
                output_from_last_node = node.run(dataframe_from_last_node)
                return
        
        _run_node(self.last_node)
        return dataframe_from_last_node

    def print_logical_plan(self):
        logical_plan = []
        def _print_op(node: LineageNode):
            if node.is_visited:
                return ""
            
            if len(node.parent) == 0:
                output_info = f"->{node.output_schema}" if node.output_schema else "\n"
                return (
                    f"{node.op_name}({node.user_instruction}): {node.input_schema}{output_info}"
                )
            
            op_info = ""
            for parent_node in node.parent:
                op_info += _print_op(parent_node)
            if op_info:
                logical_plan.append(op_info)
            
            if isinstance(node, LineageDataNode):
                return ""
            
            node.is_visited = True
            output_info = f"->{node.output_schema}" if node.output_schema else "\n"
            return (
                f"{node.op_name}({node.user_instruction}): {node.input_schema}{output_info}"
            )
        
        op_info = _print_op(self.last_node)
        if op_info:
            logical_plan.append(op_info)

        logical_plan = "=>".join(logical_plan)
        print(f"Logical Plan:\n{logical_plan}")

        def _clear_visited_flag(node: LineageNode):
            if not node.is_visited:
                return
            
            node.is_visited = False
            for parent_node in node.parent:
                _clear_visited_flag(parent_node)
            return

        _clear_visited_flag(self.last_node)
