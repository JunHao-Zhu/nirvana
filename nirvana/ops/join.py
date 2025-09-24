import asyncio
from pydantic import BaseModel
import pandas as pd
from typing import Any, List, Iterable, Tuple
from dataclasses import dataclass

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.join_prompter import JoinPrompter


def join_wrapper(
    left_data: pd.DataFrame,
    right_data: pd.DataFrame, 
    user_instruction: str,
    left_on: str,
    right_on: str,
    how: str = "inner",
    **kwargs
):
    join_op = JoinOperation()
    outputs = join_op.execute(
        left_data=left_data,
        right_data=right_data,
        user_instruction=user_instruction,
        left_on=left_on,
        right_on=right_on,
        how=how,
        **kwargs
    )
    return outputs


@dataclass
class JoinOpOutputs(BaseOpOutputs):
    joined_pairs: List[tuple] = None
    left_join_keys: List[int] = None
    right_join_keys: List[int] = None


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
        rate_limit = kwargs.get("rate_limit", 16)
        self.semaphore = asyncio.Semaphore(rate_limit)

    async def _plain_llm_execute(self, left_value: Any, right_value: Any, user_instruction: str, dtype: str, **kwargs):
        async with self.semaphore:
            full_prompt = self.prompter.generate_prompt(left_value, right_value, user_instruction, dtype)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"])
            output, total_cost = output["output"], output["cost"]
            return output, total_cost
    
    def _postprocess_join_outputs(
            self, 
            data_id_pairs: List[tuple], 
            results: Iterable[Tuple[Any, float]],
            how: str = "inner",
            left_ids: List[int] = None,
            right_ids: List[int] = None
    ):
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
        left_keys, right_keys = left_ids, right_ids
        for i, output in enumerate(join_outputs):
            if output:
                if how == "inner" or how == "left":
                    right_keys[data_id_pairs[i][1]] = left_keys[data_id_pairs[i][0]]
                else:
                    left_keys[data_id_pairs[i][0]] = right_keys[data_id_pairs[i][1]]
                joined_pairs.append(data_id_pairs[i])

        return joined_pairs, left_keys, right_keys, costs

    async def execute(
            self, 
            left_data: pd.DataFrame,
            right_data: pd.DataFrame,
            user_instruction: str,
            left_on: str,
            right_on: str,
            how: str = "inner",
            *args, 
            **kwargs
    ):
        assert left_data[left_on].dtype == right_data[right_on].dtype, (
            "Data types of columns to join must be the same."
        )
        if isinstance(left_data[left_on].dtype, ImageDtype):
            dtype = "image"
        else:
            dtype = "str"
        
        left_ids = left_data[left_on].index
        right_ids = right_data[right_on].index
        data_id_pairs = [(left_id, right_id) for left_id in left_ids for right_id in right_ids]

        tasks = []
        for left_id, right_id in data_id_pairs:
            left_value = left_data.loc[left_id, left_on]
            right_value = right_data.loc[right_id, right_on]
            tasks.append(asyncio.create_task(self._plain_llm_execute(left_value, right_value, user_instruction, dtype)))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        joined_pairs, left_join_keys, right_join_keys, token_cost = (
            self._postprocess_join_outputs(
                data_id_pairs, results, how, list(range(len(left_ids))), list(range(len(right_ids)))
            )
        )
        return JoinOpOutputs(
            output=joined_pairs,
            left_join_keys=left_join_keys,
            right_join_keys=right_join_keys,
            cost=token_cost,
        )
