import asyncio
import pandas as pd
from typing import Any, List, Iterable, Tuple
from dataclasses import dataclass

from mahjong.dataframe.arrays.image import ImageDtype
from mahjong.ops.base import BaseOpOutputs, BaseOperation
from mahjong.ops.prompt_templates.join_prompter import JoinPrompter


def join_wrapper(
    left_data: pd.DataFrame,
    right_data: pd.DataFrame, 
    user_instruction: str,
    left_column: str,
    right_column: str, 
    **kwargs
):
    join_op = JoinOperation()
    outputs = join_op.execute(
        left_data=left_data,
        right_data=right_data,
        user_instruction=user_instruction,
        left_column=left_column,
        right_column=right_column,
        strategy=None,
        **kwargs
    )
    return outputs


@dataclass
class JoinOpOutputs(BaseOpOutputs):
    output: Iterable[bool] = None
    joined_pairs: List[tuple] = None


class JoinOperation(BaseOperation):
    """
    Join operator: Join values of two columns against a specific user's instruction.

    TODO: test the join operator
    """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("join", *args, **kwargs)
        self.prompter = JoinPrompter()

    async def _plain_llm_execute(self, left_value: Any, right_value: Any, user_instruction: str, dtype: str, **kwargs):
        async with self.semaphore:
            full_prompt = self.prompter.generate_prompt(left_value, right_value, user_instruction, dtype)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"])
            output, total_cost += output["output"], output["cost"]
            return output, total_cost
    
    def _postprocess_join_outputs(self, data_id_pairs: List[tuple], results: Iterable[Tuple[Any, float]]):
        join_outputs, costs = [], 0.0
        for output, cost in results:
            if output is None:
                join_outputs.append(False)
                continue
            if isinstance(output, bool):
                join_outputs.append(output)
                continue
            if "true" in output.lower():
                join_outputs.append(True)
            elif "false" in output.lower():
                join_outputs.append(False)
            costs += cost

        joined_pairs = []
        for i, output in enumerate(join_outputs):
            if output:
                joined_pairs.append(data_id_pairs[i])

        return joined_pairs, join_outputs, costs

    async def execute(
            self, 
            left_data: pd.DataFrame,
            right_data: pd.DataFrame,
            user_instruction: str,
            left_column: str,
            right_column: str,
            how: str = "left",
            strategy: str = None,
            *args, 
            **kwargs
    ):
        assert left_data[left_column].dtype == right_data[right_column].dtype, (
            "Data types of columns to join must be the same."
        )
        if isinstance(left_data[left_column].dtype, ImageDtype):
            dtype = "image"
        else:
            dtype = "str"
        
        left_ids = left_data[left_column].index
        right_ids = right_data[right_column].index
        data_id_pairs = [(left_id, right_id) for left_id in left_ids for right_id in right_ids]

        tasks = []
        for left_id, right_id in data_id_pairs:
            left_value = left_data.loc[left_id, left_column]
            right_value = right_data.loc[right_id, right_column]
            tasks.append(asyncio.create_task(self._plain_llm_execute(left_value, right_value, user_instruction, dtype)))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        joined_pairs, join_outputs, token_cost = self._postprocess_join_outputs(data_id_pairs, results)
        return JoinOpOutputs(
            output=join_outputs,
            joined_pairs=joined_pairs,
            cost=token_cost,
        )
