"""
Record the OP lineage (operator and its user instruction) for optimizing operator orchestration.
"""

import pandas as pd

from mahjong.lineage.abstractions import LineageNode, LineageDataNode, LineageOpNode
from mahjong.lineage.plan_rewrite import OpFusion, FilterPushdown


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
        def _optimize_node(node: LineageNode):
            self.print_logical_plan()
            if isinstance(node, LineageDataNode):
                for p in node.parent:
                    _optimize_node(p)
                return
            
            if len(node.parent) == 0:
                return
            
            # apply op fusion
            optimized_node = OpFusion.rewrite_op(node)
            if optimized_node is not None:
                _optimize_node(optimized_node)
                return

            # apply filter pushdown
            optimized_node = FilterPushdown.rewrite_op(node)
            if optimized_node is not None:
                _optimize_node(optimized_node)
                return

            for p in node.parent:
                _optimize_node(p)
        
        _optimize_node(self.last_node)

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
                output_schema_info = f"{node.output_column}" if node.output_column else "Bool"
                schema_info = f"[{node.input_column}]->[{output_schema_info}]"
                return (
                    f"{node.op_name}: {schema_info} ({node.user_instruction})"
                )
            
            op_info = ""
            for parent_node in node.parent:
                op_info += _print_op(parent_node)
            if op_info:
                logical_plan.append(op_info)
            
            if isinstance(node, LineageDataNode):
                return ""
            
            node.is_visited = True
            output_schema_info = f"{node.output_column}" if node.output_column else "Bool"
            schema_info = f"[{node.input_column}]->[{output_schema_info}]"
            return (
                f"{node.op_name}: {schema_info} ({node.user_instruction})"
            )
        
        op_info = _print_op(self.last_node)
        if op_info:
            logical_plan.append(op_info)

        logical_plan = "=>\n".join(logical_plan)
        print(f"Logical Plan:\n{logical_plan}")

        def _clear_visited_flag(node: LineageNode):
            node.is_visited = False
            for parent_node in node.parent:
                _clear_visited_flag(parent_node)
            return

        _clear_visited_flag(self.last_node)
