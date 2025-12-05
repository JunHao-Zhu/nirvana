import asyncio
import functools
import pandas as pd
from typing import Any, Iterable, Callable
from dataclasses import dataclass, field

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
    join_op = JoinOperation(
        user_instruction=user_instruction,
        left_on=[left_on],
        right_on=[right_on],
        how=how,
        **kwargs
    )
    outputs = join_op.execute(
        left_data=left_data,
        right_data=right_data,
        **kwargs
    )
    return outputs


@dataclass
class JoinOpOutputs(BaseOpOutputs):
    joined_pairs: list[tuple] = field(default_factory=list)
    left_join_keys: list[int] = field(default_factory=list)
    right_join_keys: list[int] = field(default_factory=list)


class JoinOperation(BaseOperation):
    """
    Join operator: Join values of two columns against a specific user's instruction.

    TODO: test the join operator
    """
    implementation_options = ["nest", "block"]
    
    def __init__(
        self,
        user_instruction: str = "",
        left_on: list[str] = [],
        right_on: list[str] = [],
        how: str = "inner",
        **kwargs,
    ):
        super().__init__(
            op_name="join", 
            user_instruction=user_instruction,
            **kwargs
        )
        self.prompter = JoinPrompter()
        self.left_on = left_on
        self.right_on = right_on
        self.how = how

    @property
    def dependencies(self) -> list[str]:
        return self.left_on + self.right_on

    @property
    def generated_fields(self) -> list[str]:
        return []
    
    @property
    def op_kwargs(self) -> dict:
        kwargs = super().op_kwargs
        kwargs["left_on"] = self.left_on
        kwargs["right_on"] = self.right_on
        kwargs["how"] = self.how
        return kwargs
    
    def _prepare_nested_join_pairs(self, left_values, right_values):
        left_ids = left_values.index
        right_ids = right_values.index
        data_id_pairs = [(left_id, right_id) for left_id in left_ids for right_id in right_ids]
        return data_id_pairs, left_ids.tolist(), right_ids.tolist()
    
    async def _execute_by_func(self, left_value: Any, right_value: Any, user_instruction: str, func: Callable, llm_call: Callable, **kwargs):
        try:
            join_result = func(left_value, right_value)
            output = {"output": join_result, "cost": 0.0}
            return output
        except Exception as e:
            return await llm_call(left_value, right_value, user_instruction)

    async def _pairwise_evaluate(self, left_value: Any, right_value: Any, user_instruction: str, left_dtype: str, right_dtype: str, **kwargs):
        async with self.semaphore:
            if left_dtype == "str":
                left_value = f"{self.left_on[0]}: {str(left_value)}"
            if right_dtype == "str":
                right_value = f"{self.right_on[0]}: {str(right_value)}"
            full_prompt = self.prompter.generate_nested_join_prompt(left_value, right_value, user_instruction, left_dtype, right_dtype)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return output
    
    def _postprocess_nested_join_outputs(
        self, 
        data_id_pairs: list[tuple], 
        results: Iterable[dict],
        how: str = "inner",
        left_ids: list[int] = None,
        right_ids: list[int] = None
    ):
        join_outputs, total_cost = [], 0.0
        for result in results:
            if result is None:
                join_outputs.append(False)
                continue
            
            output, cost = result["output"], result["cost"]
            if output is None:
                join_outputs.append(False)
            elif isinstance(output, bool):
                join_outputs.append(output)
            elif "true" in output.lower():
                join_outputs.append(True)
            elif "false" in output.lower():
                join_outputs.append(False)
            total_cost += cost

        joined_pairs = []
        left_keys, right_keys = left_ids, right_ids
        for i, output in enumerate(join_outputs):
            id_in_left, id_in_right = i // len(right_ids), i % len(right_ids)
            if output:
                if how == "inner" or how == "left":
                    right_keys[id_in_right] = left_keys[id_in_left]
                else:
                    left_keys[id_in_left] = right_keys[id_in_right]
                joined_pairs.append(data_id_pairs[i])

        return joined_pairs, left_keys, right_keys, total_cost
        
    async def _nested_join(
        self, 
        left_data: pd.DataFrame, 
        right_data: pd.DataFrame, 
        user_instruction: str, 
        left_dtype: str, 
        right_dtype: str, 
        **kwargs
    ):
        execution_func = functools.partial(self._pairwise_evaluate, left_dtype=left_dtype, right_dtype=right_dtype, **kwargs)
        # Prepare candidate pairs
        data_id_pairs, left_ids, right_ids = self._prepare_nested_join_pairs(left_data, right_data)
        tasks = []
        for left_id, right_id in data_id_pairs:
            left_value = left_data.loc[left_id, self.left_on[0]]
            right_value = right_data.loc[right_id, self.right_on[0]]
            if pd.isna(left_value) or pd.isna(right_value):
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
            elif self.has_udf():
                tasks.append(asyncio.create_task(self._execute_by_func(left_value, right_value, user_instruction, self.tool, execution_func)))
            else:
                tasks.append(asyncio.create_task(execution_func(left_value, right_value, user_instruction)))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        joined_pairs, left_join_keys, right_join_keys, token_cost = self._postprocess_nested_join_outputs(
            data_id_pairs, results, self.how, left_ids, right_ids
        )
        return JoinOpOutputs(
            output=joined_pairs,
            left_join_keys=left_join_keys,
            right_join_keys=right_join_keys,
            cost=token_cost,
        )
    
    def _prepare_join_batches(
        self, 
        left_values: pd.Series, 
        right_values: pd.Series, 
        batch_size: int,
    ):
        # Prepare left batches
        left_batches, left_keys = [], []
        start_idx = 0
        while start_idx < len(left_values):
            left_batches.append(left_values.loc[start_idx:start_idx+batch_size-1].tolist())
            left_keys.append(left_values.index[start_idx:start_idx+batch_size-1].tolist())
            start_idx += batch_size
        
        # Prepare right batches
        right_batches, right_keys = [], []
        start_idx = 0
        while start_idx < len(right_values):
            right_batches.append(right_values.loc[start_idx:start_idx+batch_size-1].tolist())
            right_keys.append(right_values.index[start_idx:start_idx+batch_size-1].tolist())
            start_idx += batch_size
        
        return left_batches, left_keys, right_batches, right_keys

    async def _batchwise_evaluate(self, left_batch: list, right_batch: list, user_instruction: str, left_dtype: str, right_dtype: str, **kwargs):
        async with self.semaphore:
            full_prompt = self.prompter.generate_batch_join_prompt(left_batch, right_batch, user_instruction, left_dtype, right_dtype)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return output
        
    def _postprocess_block_join_outputs(
        self,
        results: Iterable[dict],
        how: str,
        left_keys_in_batches: list[list],
        right_keys_in_batches: list[list],
    ):
        def _extract_pairs_from_llm_output(output: str, keys_in_left_batch, keys_in_right_batch):
            if output is None or output == "":
                return []
            pairs_str = output.split(',')
            joined_pairs = []
            for pair_str in pairs_str:
                pair_str = pair_str.strip()
                left_ref, right_ref = pair_str.split('-')
                left_idx, right_idx = keys_in_left_batch[int(left_ref[1:])], keys_in_right_batch[int(right_ref[1:])]
                pair = (left_idx, right_idx)
                joined_pairs.append(pair)
            return joined_pairs
        
        joined_pairs, total_cost = [], 0.0
        for batch_id, result in enumerate(results):
            output, cost = result["output"], result["cost"]
            joined_pairs_per_block = _extract_pairs_from_llm_output(output, left_keys_in_batches[batch_id], right_keys_in_batches[batch_id])
            joined_pairs.extend(joined_pairs_per_block)
            total_cost += cost

        key_mapping = {}
        if how == "inner" or how == "left":
            for left_key, right_key in joined_pairs:
                key_mapping[left_key] = right_key
            return joined_pairs, list(key_mapping.keys()), list(key_mapping.values()), total_cost
        else:
            for right_key, left_key in joined_pairs:
                key_mapping[right_key] = left_key
            return joined_pairs, list(key_mapping.values()), list(key_mapping.keys()), total_cost
        
    async def _block_join(
        self,
        left_data: pd.DataFrame,
        right_data: pd.DataFrame,
        user_instruction: str,
        batch_size: int,
        left_dtype: str,
        right_dtype: str,
        **kwargs
    ):
        # Prepare batches
        left_values = left_data[self.left_on[0]].map(lambda x: f"{self.left_on}: {str(x)}") if left_dtype == "str" else left_data[self.left_on[0]]
        right_values = right_data[self.right_on[0]].map(lambda x: f"{self.right_on}: {str(x)}") if right_dtype == "str" else right_data[self.right_on[0]]
        left_batches, left_keys, right_batches, right_keys = self._prepare_join_batches(left_values, right_values, batch_size=batch_size)

        tasks = []
        for left_batch, right_batch in zip(left_batches, right_batches):
            tasks.append(asyncio.create_task(self._batchwise_evaluate(left_batch, right_batch, user_instruction, left_dtype, right_dtype, **kwargs)))
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        joined_pairs, left_join_keys, right_join_keys, token_cost = self._postprocess_block_join_outputs(
            results, self.how, left_keys, right_keys
        )
        return JoinOpOutputs(
            output=joined_pairs,
            left_join_keys=left_join_keys,
            right_join_keys=right_join_keys,
            cost=token_cost,
        )

    async def execute(
        self, 
        left_data: pd.DataFrame,
        right_data: pd.DataFrame,
        **kwargs
    ):
        assert left_data[self.left_on[0]].dtype == right_data[self.right_on[0]].dtype, (
            "Data types of columns to join must be the same."
        )
        if left_data.empty or right_data.empty:
            return JoinOpOutputs(
                output=[],
                left_join_keys=[],
                right_join_keys=[],
                cost=0.0,
            )
        
        left_dtype = "image" if isinstance(left_data[self.left_on[0]].dtype, ImageDtype) else "str"
        right_dtype = "image" if isinstance(right_data[self.right_on[0]].dtype, ImageDtype) else "str"

        if self.implementation == "nest":
            join_func = functools.partial(self._nested_join, left_dtype=left_dtype, right_dtype=right_dtype, **kwargs)
        elif self.implementation == "block":
            if self.has_udf():
                raise ValueError("The block join implementation does not support user-defined functions.")
            join_func = functools.partial(self._block_join, batch_size=kwargs.get("batch_size", 5), left_dtype=left_dtype, right_dtype=right_dtype, **kwargs)
        else:
            raise ValueError(f"The optional implementations available for join are {self.implementation_options}. Strategy {self.implementation} is not available.")
        
        return await join_func(left_data, right_data, self.user_instruction)
