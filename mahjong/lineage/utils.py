import time
import asyncio
import pandas as pd

from mahjong.lineage.abstractions import LineageNode, LineageDataNode, LineageOpNode

schema_mapping = {
    "map": "[{input_column}]->[{output_column}]",
    "filter": "[{input_column}]->[Bool]",
    "reduce": "[{input_column}]->[Aggregated Result]",
}


def collect_op_metadata(op_node: LineageOpNode, print_info: bool = False):
    op_name = op_node.op_name
    user_instr = op_node.user_instruction
    input_col = op_node.input_column
    output_col = op_node.output_column
    has_func = True if op_node.func else False
    if print_info:
        op_signature = (
            f"{schema_mapping[op_name].format(input_column=input_col, output_column=output_col)} ({user_instr})"
        )
        return op_signature
    else:
        metadata = (
            op_name, user_instr, input_col, output_col, has_func
        )
        return metadata


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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            output_from_last_node = loop.run_until_complete(node.run(execute_output["dataframe_from_last_node"]))
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
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            output_from_last_node = loop.run_until_complete(node.run(execute_output["dataframe_from_last_node"]))
            execute_output["output_from_last_node"] = output_from_last_node
            execute_output["total_token_cost"] += output_from_last_node.cost
            return
    execution_start_time = time.time()
    _run_node(last_node)
    execution_end_time = time.time()
    execution_time = execution_end_time - execution_start_time
    return execute_output["dataframe_from_last_node"], execute_output["total_token_cost"], execution_time
