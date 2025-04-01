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
    def __init__(self, op_name, user_instruction: str,):
        self.op_name = op_name
        self.op = op_mapping[op_name]()
        self.user_instruction = user_instruction

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


class LineageMixin:
    def __init__(self):
        self.last_op = None

    def add_operator(self, op_name, user_instruction):
        op_node = LineageNode(op_name, user_instruction)
        if self.last_op is None:
            self.last_op = op_node
        else:
            op_node.add_parent(self.last_op)
            self.last_op.add_child(op_node)
            self.last_op = op_node

    def optimize(self):
        pass

    def print_logic_plan(self):
        print("Logic Plan:\n")
        plan = []
        
        def _print_op(node: LineageNode):
            if len(node.parent) == 0:
                return f"({node.op_name}, {node.user_instruction})"
            
        pass
