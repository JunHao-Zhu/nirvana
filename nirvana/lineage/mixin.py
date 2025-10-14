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
    "map": "MAP({user_instruction}):[{input_column}]->[{output_column}]",
    "filter": "FILTER({user_instruction}):[{input_column}]->[Bool]",
    "reduce": "AGGR({user_instruction}):[{input_column}]->[Aggr]",
    "join": "{how}-JOIN({user_instruction}):[left[{left_on}] * right[{right_on}]",
}


def collect_op_metadata(op_node: LineageNode, print_info: bool = False):
    op_name = op_node.op_name
    op_kwargs = op_node.op_kwargs
    has_func = True if op_node.func else False
    if print_info:
        op_signature = (
            f"{schema_mapping[op_name].format(**op_kwargs)}"
        )
        return op_signature
    else:
        # Note: there might be a bug
        metadata = (
            op_name, *op_kwargs.values(), has_func
        )
        return metadata


def execute_along_lineage(leaf_node: LineageNode):
    total_token_cost = 0
    def _execute_node(node: LineageNode) -> pd.DataFrame:
        if node.left_parent:
            left_node_output = _execute_node(node.left_parent)
        if node.right_parent:
            right_node_output = _execute_node(node.right_parent)
        
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
        data_kwargs = {"output_fields": self.columns}
        node = LineageNode(op_name="scan", data_metadata=data_kwargs, datasource=self._data)
        self.leaf_node = node

    def add_operator(self, op_name: str, op_kwargs: dict, data_kwargs: dict, **kwargs):
        node = LineageNode(op_name, op_metadata=op_kwargs, data_metadata=data_kwargs)
        if op_name == "join":
            node.set_left_parent(self.leaf_node)
            node.set_right_parent(kwargs["other"].leaf_node)
            self.leaf_node = node
        else:
            node.set_left_parent(self.leaf_node)
            self.leaf_node = node

    def create_plan_optimizer(self, config: OptimizeConfig = None):
        self.optimizer = PlanOptimizer(config)

    def execute(self):
        return execute_along_lineage(self.leaf_node)
        
    def print_lineage_graph(self):
        lineage_graph_strings = []
        op_strings_in_same_hop = []
        node_queue = deque([self.leaf_node])

        while node_queue:
            node = node_queue.popleft()
            if node is None:
                lineage_graph_strings.append(op_strings_in_same_hop)
                op_strings_in_same_hop = []
                continue

            op_info = collect_op_metadata(node, print_info=True)
            op_strings_in_same_hop.append(op_info)

            node_queue.append(node.left_parent)
            if node.right_parent:
                node_queue.append(node.right_parent)
            node_queue.append(None)

        stringified_lineage_graph = ""
        while lineage_graph_strings:
            ops_info = lineage_graph_strings.pop()
            ops_info_string = ""
            for op_info in ops_info:
                if len(op_info) > 20:
                    op_info = f"{op_info[:17] + '...':<20}\t"
                else:
                    op_info = f"{op_info:<20}\t"
                ops_info_string += op_info
            divider = "{|:<20}\t" * len(ops_info)
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
            if node.left_parent:
                _delete_node(node.left_parent)
            if node.right_parent:
                _delete_node(node.right_parent)
            del node
            return
        _delete_node(temp_node)
        del temp_node
