"""
Record the OP lineage (operator and its user instruction) for optimizing operator orchestration.
"""
import copy
import time
import asyncio
import pandas as pd
from collections import deque

from nirvana.lineage.abstractions import LineageNode
from nirvana.optim.optimizer import PlanOptimizer, OptimizeConfig

schema_mapping = {
    "scan": "SCAN({source}):[]->[{output_columns}]",
    "map": "MAP({user_instruction}):[{input_column}]->[{output_column}]",
    "filter": "FILTER({user_instruction}):[{input_column}]->[Bool]",
    "rank": "RANK({user_instruction}):[{input_column}]->[Rank]",
    "reduce": "AGGR({user_instruction}):[{input_column}]->[Aggr]",
    "join": "{how}-JOIN({user_instruction}):[left[{left_on}] * right[{right_on}]",
}


def collect_op_metadata(op_node: LineageNode, max_instruction_print_length: int = 50):
    op_name = op_node.op_name
    op_kwargs = op_node.operator.op_kwargs
    user_instruction = op_kwargs["user_instruction"][:max_instruction_print_length]
    if op_name == "map":
        return (
            schema_mapping[op_name].format(user_instruction=user_instruction, input_column=op_kwargs['input_columns'][0], output_column=op_kwargs['output_columns'][0])
        )
    elif op_name in ["filter", "rank", "reduce"]:
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
    total_token_cost = 0
    def _execute_node(node: LineageNode) -> pd.DataFrame:
        if node.left_child:
            left_node_output = _execute_node(node.left_child)
        if node.right_child:
            right_node_output = _execute_node(node.right_child)
        
        if node.op_name == "scan":
            node_output = asyncio.run(node.run())
            total_token_cost += node_output.cost
            return node_output.output
        
        elif node.op_name == "join":
            node_output = asyncio.run(node.run([left_node_output, right_node_output]))
            total_token_cost += node_output.cost
            return node_output.output
        
        else:
            node_output = asyncio.run(node.run(left_node_output))
            total_token_cost += node_output.cost
            return node_output
    
    execution_start_time = time.time()
    output_from_lineage = _execute_node(leaf_node)
    execution_end_time = time.time()
    execution_time = execution_end_time - execution_start_time
    return output_from_lineage, total_token_cost, execution_time


class LineageMixin:
    def initialize(self):
        op_kwargs = {"source": "dataframe", "output_columns": self.columns}
        data_kwargs = {"left_input_fields": [], "right_input_fields": [], "output_fields": self.columns}
        node = LineageNode(op_name="scan", op_kwargs=op_kwargs, node_fields=data_kwargs, datasource=self._data)
        self.leaf_node = node

    def add_operator(self, op_name: str, op_kwargs: dict, data_kwargs: dict, **kwargs):
        node = LineageNode(op_name, op_kwargs=op_kwargs, node_fields=data_kwargs)
        if op_name == "join":
            node.set_left_child(self.leaf_node)
            node.set_right_child(kwargs["other"].leaf_node)
            self.leaf_node = node
        else:
            node.set_left_child(self.leaf_node)
            self.leaf_node = node

    def create_plan_optimizer(self, config: OptimizeConfig = None):
        self.optimizer = PlanOptimizer(config)

    def execute(self):
        return execute_along_lineage(self.leaf_node)
        
    def print_lineage_graph(self, op_signature_width: int = 100, max_instruction_print_length: int = 50):
        lineage_graph_strings = []
        op_strings_in_same_hop = []
        node_queue = deque([self.leaf_node])

        while node_queue:
            node = node_queue.popleft()
            if node is None:
                lineage_graph_strings.append(op_strings_in_same_hop)
                op_strings_in_same_hop = []
                continue

            op_info = collect_op_metadata(node, max_instruction_print_length)
            op_strings_in_same_hop.append(op_info)

            node_queue.append(None)
            node_queue.append(node.left_child)
            if node.right_child:
                node_queue.append(node.right_child)

        stringified_lineage_graph = ""
        while lineage_graph_strings:
            ops_info = lineage_graph_strings.pop()
            ops_info_string = ""
            for op_info in ops_info:
                op_info = f"{op_info:<{op_signature_width}}"
                ops_info_string += op_info
            divider = f"{'|':<{op_signature_width}}\t" * len(ops_info)
            stringified_lineage_graph += ops_info_string.strip() + "\n"
            stringified_lineage_graph += divider + "\n"

        print(f"Lineage Graph:\n{stringified_lineage_graph}")

    def clear_lineage_graph(self):
        self.optimizer.clear()
        temp_node = copy.copy(self.leaf_node)
        self.leaf_node = None
        # See join left and right tables in data lineage, 
        # empty_lineage will delete all nodes along two upstream sub-lineages
        # So put a note here if there is a bug when deleting nodes
        def _delete_node(node: LineageNode):
            if node.left_child:
                _delete_node(node.left_child)
            if node.right_child:
                _delete_node(node.right_child)
            del node
            return
        _delete_node(temp_node)
        del temp_node
