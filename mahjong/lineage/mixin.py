"""
Record the OP lineage (operator and its user instruction) for optimizing operator orchestration.
"""
import copy
import time
import pandas as pd

from mahjong.lineage.abstractions import LineageNode, LineageDataNode, LineageOpNode
from mahjong.lineage.plan_rewrite import OpFusion, FilterPushdown


def execute_plan(last_node: LineageNode, input_data: pd.DataFrame):
    execute_output = {
        "dataframe_from_last_node": input_data.copy(),
        "output_from_last_node": None,
        "total_token_cost": 0,
    }
    def _run_node(node: LineageNode):
        # if the node is the first operator, run it on input data
        if len(node.parent) == 0:
            assert isinstance(node, LineageOpNode), "The first node should be an operator."
            output_from_last_node = node.run(execute_output["dataframe_from_last_node"])
            execute_output["output_from_last_node"] = output_from_last_node
            execute_output["total_token_cost"] += output_from_last_node.cost
            return
        # run the parent nodes
        for parent_node in node.parent:
            _run_node(parent_node)

        if isinstance(node, LineageDataNode):
            dataframe_from_last_node = node.run(execute_output["dataframe_from_last_node"], execute_output["output_from_last_node"])
            execute_output["dataframe_from_last_node"] = dataframe_from_last_node
            return

        if isinstance(node, LineageOpNode):
            output_from_last_node = node.run(execute_output["dataframe_from_last_node"])
            execute_output["output_from_last_node"] = output_from_last_node
            execute_output["total_token_cost"] += output_from_last_node.cost
            return
    execution_start_time = time.time()
    _run_node(last_node)
    execution_end_time = time.time()
    execution_time = execution_end_time - execution_start_time
    return execute_output["dataframe_from_last_node"], execute_output["total_token_cost"], execution_time


class LineageMixin:

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
        if self.last_node is None:
            raise RuntimeError("No operations have been added to the DataFrame.")
        self.last_node = self.optimizer.optimize(self.last_node, "df", self.columns)

    # def optimize(self):
    #     def _optimize_node(node: LineageNode):
    #         self.print_logical_plan()
    #         if isinstance(node, LineageDataNode):
    #             for p in node.parent:
    #                 _optimize_node(p)
    #             return
            
    #         if len(node.parent) == 0:
    #             return
            
    #         # apply op fusion
    #         optimized_node = OpFusion.rewrite_op(node)
    #         if optimized_node is not None:
    #             _optimize_node(optimized_node)
    #             return

    #         # apply filter pushdown
    #         optimized_node = FilterPushdown.rewrite_op(node)
    #         if optimized_node is not None:
    #             _optimize_node(optimized_node)
    #             return

    #         for p in node.parent:
    #             _optimize_node(p)
        
    #     _optimize_node(self.last_node)

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

        self._clear_visited_flag(self.last_node)

    def _clear_visited_flag(self, node: LineageNode):
        node.is_visited = False
        for parent_node in node.parent:
            self._clear_visited_flag(parent_node)
        return

    def empty_lineage(self):
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
