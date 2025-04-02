"""
Record the OP lineage (operator and its user instruction) for optimizing operator orchestration.
"""

from typing import List

from mahjong.ops.map import MapOperation
from mahjong.ops.filter import FilterOperation
from mahjong.ops.reduce import ReduceOperation


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


class LineageOpNode(LineageNode):
    def __init__(self, op_name, user_instruction: str,):
        super().__init__()
        self.op_name = op_name
        self.op = op_mapping[op_name]()
        self.user_instruction = user_instruction
    
    def add_child(self, node: LineageNode):
        self._child.append(node)

    def add_parent(self, node: LineageNode):
        self._parent.append(node)


class LineageMixin:
    def __init__(self):
        self.last_op = None

    def add_operator(self, op_name, user_instruction):
        op_node = LineageOpNode(op_name, user_instruction)
        if self.last_op is None:
            self.last_op = op_node
        else:
            op_node.add_parent(self.last_op)
            self.last_op.add_child(op_node)
            self.last_op = op_node

    def optimize(self):
        pass

    def print_logical_plan(self):
        print("Logical Plan:")

        logical_plan = []
        def _print_op(node: LineageNode):
            if node.is_visited:
                return ""
            
            if len(node.parent) == 0:
                return f"({node.op_name}, {node.user_instruction})\n"
            
            op_info = ""
            for parent_node in node.parent:
                op_info += _print_op(parent_node)
            if op_info:
                logical_plan.append(op_info)
            
            node.is_visited = True
            return f"({node.op_name}, {node.user_instruction})\n"
        
        op_info = _print_op(self.last_op)
        if op_info:
            logical_plan.append(op_info)

        logical_plan = "=>".join(logical_plan)
        print(logical_plan)

        def _clear_visited_flag(node: LineageNode):
            if not node.is_visited:
                return
            
            node.is_visited = False
            for parent_node in node.parent:
                _clear_visited_flag(parent_node)
            return

        _clear_visited_flag(self.last_op)
