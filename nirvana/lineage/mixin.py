"""
Record the OP lineage (operator and its user instruction) for optimizing operator orchestration.
"""
import copy
import time
import asyncio
from collections import deque

from nirvana.lineage.abstractions import LineageNode, execute_node, collect_op_metadata
from nirvana.optim.optimizer import PlanOptimizer, OptimizeConfig

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
        execution_start_time = time.time()
        dataframe_from_node, token_cost = asyncio.run(execute_node(self.leaf_node))
        execution_end_time = time.time()
        execution_time = execution_end_time - execution_start_time
        return dataframe_from_node, token_cost, execution_time
    
    def print_lineage_graph(self, op_signature_width: int = 512, max_instruction_print_length: int = 256):
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
        def _delete_node(node: LineageNode):
            if node.left_child:
                _delete_node(node.left_child)
            if node.right_child:
                _delete_node(node.right_child)
            del node
            return
        
        if self.optimizer:
            self.optimizer.clear()
        if self.leaf_node:
            temp_node = copy.copy(self.leaf_node)
            self.leaf_node = None
            # See join left and right tables in data lineage, 
            # empty_lineage will delete all nodes along two upstream sub-lineages
            # So put a note here if there is a bug when deleting nodes
            _delete_node(temp_node)
            del temp_node
