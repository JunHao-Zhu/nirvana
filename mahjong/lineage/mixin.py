"""
Record the OP lineage (operator and its user instruction) for optimizing operator orchestration.
"""
import copy
import pandas as pd

from mahjong.lineage.abstractions import LineageNode, LineageDataNode, LineageOpNode
from mahjong.lineage.utils import execute_plan, collect_op_metadata


class LineageMixin:

    def add_operator(self, op_name, user_instruction, input_column, output_column=None, fields=None):
        op_node = LineageOpNode(
            op_name, user_instruction, None, input_column, output_column
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
        if self.last_node is None:
            raise RuntimeError("No operations have been added to the DataFrame.")
        self.last_node = self.optimizer.optimize(self.last_node, "df", self.columns)

    def execute(self, input_data: pd.DataFrame = None):
        if input_data is None:
            return execute_plan(self.last_node, self._data)
        return execute_plan(self.last_node, input_data)

    def print_logical_plan(self):
        if self.last_node is None:
            print("No operations have been added to the DataFrame.")
            return
        logical_plan = []
        def _print_op(node: LineageNode):
            if node.is_visited:
                return ""
            
            if len(node.parent) == 0:
                op_info = collect_op_metadata(node, print_info=True)
                return op_info
            
            op_info = ""
            for parent_node in node.parent:
                op_info += _print_op(parent_node)
            if op_info:
                logical_plan.append(op_info)
            
            if isinstance(node, LineageDataNode):
                return ""
            
            node.is_visited = True
            op_info = collect_op_metadata(node, print_info=True)
            return op_info
        
        op_info = _print_op(self.last_node)
        if op_info:
            logical_plan.append(op_info)

        logical_plan = "=>\n".join(logical_plan)
        print(f"Logical Plan:\n{logical_plan}")

        self._clear_visited_flag(self.last_node)

    def _clear_visited_flag(self, node: LineageNode):
        node.is_visited = False
        for parent_node in node.parent:
            self._clear_visited_flag(parent_node)
        return

    def empty_lineage(self):
        self.optimizer.clear()
        temp_node = copy.copy(self.last_node)
        self.last_node = None
        def _delete_node(node: LineageNode):
            if len(node.parent) == 0:
                del node
                return
            for p in node.parent:
                _delete_node(p)
            del node
            return
        _delete_node(temp_node)
        del temp_node
