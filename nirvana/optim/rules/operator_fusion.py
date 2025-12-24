import asyncio
from enum import Enum

from nirvana.executors.tools import FunctionCallTool
from nirvana.executors.llm_backbone import LLMClient
from nirvana.lineage.abstractions import LineageNode


class FusionType(Enum):
    SAME_TYPE = 1
    MAP_FILTER = 2


class OperatorFusion:
    """
    Rule-based operator fusion on the lineage tree.

    Two main patterns are supported:
    1) Merge multiple same-type operators (e.g., map-map or filter-filter) that work
       on the same column into a single operator whose schema (e.g., user_instruction)
       preserves all original instructions.
    2) Merge a filter and a map that work on the same column into a single map
       operator whose instruction is rewritten to perform both the filtering
       and mapping; a trailing filter with a UDF is then added to remove None results.
    """

    instruction_rewrite_prompt = """There are two semantic data analytics operations evaluating some instructions on data.
You are tasked with merging these instructions across operations into one instruction for a merged operator. There are two criteria:
1. The generated instruction must include all key requirements in those instructions.
2. If instructions across a filter and a map are to merge, the new instruction must include a mandatory requirement that if data don't pass the filter condition, return None for each field as the map result; otherwise, perform data transformation as normal.

The operations and their instructions are follows:
{instruction}

Output the merged instruction concisely in the following format.
<instruction> The merged instruction </instruction>
"""

    @classmethod
    def match_pattern(cls, node_a: LineageNode, node_b: LineageNode, pattern: FusionType) -> bool:
        if node_a.operator.has_udf() or node_b.operator.has_udf():
            return False
        
        if pattern == FusionType.SAME_TYPE:
            return node_a.op_name == node_b.op_name
        else:
            return (
                (node_a.op_name in {"map", "filter"} and node_b.op_name in {"map", "filter"}) and 
                (node_a.operator.dependencies == node_b.operator.dependencies)
            )

    @classmethod
    def transform(cls, node: LineageNode, rewriter: LLMClient) -> LineageNode:
        if node.op_name == "scan":
            return node
        
        if node.op_name == "join":
            left_child = cls.transform(node.left_child)
            right_child = cls.transform(node.right_child)
            if left_child:
                node.set_left_child(left_child)
            if right_child:
                node.set_right_child(right_child)
            return node
        
        if node.op_name in {"rank", "reduce"}:
            child = cls.transform(node.left_child)
            node.set_left_child(child)
            return node
        
        child = cls.transform(node.left_child)
        node.set_left_child(child)
        if cls.match_pattern(node, child, FusionType.SAME_TYPE):
            node, cost = cls._merge_same_type_operators(node, child, rewriter)
        if cls.match_pattern(node, child, FusionType.MAP_FILTER):
            node, cost = cls._merge_map_and_filter(node, child, rewriter)
        return node
    
    @classmethod
    def _merge_same_type_operators(
        cls,
        node: LineageNode,
        node_to_merge: LineageNode,
        rewriter: LLMClient,
    ) -> tuple[LineageNode, float]:
        """
        Merge operators that share a same type as `node` and depend on the same column.
        """
        # Prepare new instruction for the fused operator
        combined_instruction = (
            f"{node.op_name} 1: {node.operator.user_instruction}\n"
            f"{node.op_name} 2: {node_to_merge.operator.user_instruction}"
        )
        prompt = cls.instruction_rewrite_prompt.format(instruction=combined_instruction)
        llm_output = asyncio.run(rewriter(prompt, parse_tags=True, tags=["instruction"]))
        new_instruction, cost = llm_output["instruction"], llm_output["cost"]

        # Prepare new operator
        op_name = node.op_name
        op_kwargs: dict = node.operator.op_kwargs
        op_kwargs["user_instruction"] = new_instruction
        dependencies = set(node.operator.dependencies + node_to_merge.operator.dependencies)
        op_kwargs["input_columns"] = list(dependencies)

        if op_name == "map":
            generated_fields = set(node.operator.generated_fields + node_to_merge.operator.generated_fields)
            op_kwargs["output_columns"] = list(generated_fields)
        node_fields = {
            "left_input_fields": node_to_merge.node_fields.left_input_fields,
            "right_input_fields": [],
            "output_fields": list(
                set(node_to_merge.node_fields.left_input_fields + op_kwargs.get("output_columns", []))
            )
        }
        new_op = LineageNode(
            op_name=op_name,
            op_kwargs=op_kwargs,
            node_fields=node_fields,
        )
        new_op.set_left_child(node_to_merge.left_child)
        return new_op, cost

    @classmethod
    def _merge_map_and_filter(
        cls,
        node: LineageNode,
        node_to_merge: LineageNode,
        rewriter: LLMClient,
    ) -> tuple[LineageNode, float]:
        """
        Merge a map and a filter into a single map followed by a code filter
        with a function `lambda x: x is not None`. 
        I.e., `Map + Filter -> Map + Filter(lambda x: x is not None)`
        or `Filter + Map -> Map + Filter(lambda x: x is not None)`
        """
        # Prepare new instruction for the fused operator
        combined_instruction = (
            f"{node.op_name} 1: {node.operator.user_instruction}\n"
            f"{node_to_merge.op_name} 1: {node_to_merge.operator.user_instruction}"
        )
        prompt = cls.instruction_rewrite_prompt.format(instruction=combined_instruction)
        llm_output = asyncio.run(rewriter(prompt, parse_tags=True, tags=["instruction"]))
        new_instruction, cost = llm_output["instruction"], llm_output["cost"]

        # Prepare new map operator
        op_kwargs = node.operator.op_kwargs if node.op_name == "map" else node_to_merge.operator.op_kwargs
        op_kwargs["user_instruction"] = new_instruction
        dependencies = set(node.operator.dependencies + node_to_merge.operator.dependencies)
        op_kwargs["input_columns"] = list(dependencies)

        generated_fields = node.operator.generated_fields if node.op_name == "map" else node_to_merge.operator.generated_fields
        op_kwargs["output_columns"] = list(generated_fields)
        node_fields = {
            "left_input_fields": node_to_merge.node_fields.left_input_fields,
            "right_input_fields": [],
            "output_fields": list(
                set(node_to_merge.node_fields.left_input_fields + op_kwargs.get("output_columns", []))
            )
        }
        new_map_node = LineageNode(
            op_name="map",
            op_kwargs=op_kwargs,
            node_fields=node_fields,
        )
        new_map_node.set_left_child(node_to_merge.left_child)

        filter_kwargs = {
            "user_instruction": f"Fields {op_kwargs['output_columns']} contain None",
            "input_columns": op_kwargs["output_columns"],
            "tool": FunctionCallTool.from_function(func=lambda x: x.hasnans),
        }
        node_fields = {
            "left_input_fields": new_map_node.node_fields.output_fields,
            "right_input_fields": [],
            "output_fields": new_map_node.node_fields.output_fields,
        }
        code_filter = LineageNode(
            op_name="filter",
            op_kwargs=filter_kwargs,
            node_fields=node_fields
        )
        code_filter.set_left_child(new_map_node)
        return code_filter, cost
