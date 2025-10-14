import time
import asyncio
import pandas as pd

from nirvana.lineage.abstractions import LineageNode, LineageDataNode, LineageOpNode

schema_mapping = {
    "map": "MAP({user_instruction}):[{input_column}]->[{output_column}]",
    "filter": "FILTER({user_instruction}):[{input_column}]->[Bool]",
    "reduce": "AGGR({user_instruction}):[{input_column}]->[Aggr]",
    "join": "{how}-JOIN({user_instruction}):[left[{left_on}] * right[{right_on}]",
}


def collect_op_metadata(op_node: LineageOpNode, print_info: bool = False):
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


def execute_plan(last_node: LineageNode, input_data: pd.DataFrame):
    execute_output = {
        "dataframe_from_last_node": None,
        "output_from_last_node": None,
        "total_token_cost": 0,
    }
    def _run_node(node: LineageNode):
        # The first node has to be a data node where data are scanned from in-/out-memory data stores
        if len(node.parent) == 0:
            assert isinstance(node, LineageDataNode), "The first node should include data."
            execute_output["dataframe_from_last_node"] = node.data
            return
        # run the parent nodes
        for parent_node in node.parent:
            _run_node(parent_node)

        if isinstance(node, LineageDataNode):
            dataframe_from_last_node = node.run(execute_output["dataframe_from_last_node"], execute_output["output_from_last_node"])
            execute_output["dataframe_from_last_node"] = dataframe_from_last_node
            return

        if isinstance(node, LineageOpNode):
            output_from_last_node = asyncio.run(node.run(execute_output["dataframe_from_last_node"]))
            execute_output["output_from_last_node"] = output_from_last_node
            execute_output["total_token_cost"] += output_from_last_node.cost
            return
    execution_start_time = time.time()
    _run_node(last_node)
    execution_end_time = time.time()
    execution_time = execution_end_time - execution_start_time
    return execute_output["dataframe_from_last_node"], execute_output["total_token_cost"], execution_time
