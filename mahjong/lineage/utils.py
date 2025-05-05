import time
import pandas as pd

from mahjong.lineage.abstractions import LineageNode, LineageDataNode, LineageOpNode


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
