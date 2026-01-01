import warnings

import asyncio
import functools
import pandas as pd
from typing import Any, Iterable, Callable, Literal
from dataclasses import dataclass, field

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.executors.tools import BaseTool, FunctionCallTool
from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.join_prompter import JoinPrompter


def join_wrapper(
    left_data: pd.DataFrame,
    right_data: pd.DataFrame, 
    user_instruction: str,
    left_on: str,
    right_on: str,
    how: str = "inner",
    context: list[dict] | str | None = None,
    model: str | None = None,
    func: Callable = None,
    strategy: Literal["nest", "block"] = "nest",
    limit: int | None = None,
    rate_limit: int = 16,
    assertions: list[Callable] | None = [],
    batch_size: int = 5,
    **kwargs
):
    """
    A function wrapper for join operation

    Args:
        left_data (pd.DataFrame): Left dataframe
        right_data (pd.DataFrame): Right dataframe
        user_instruction (str): User instruction
        left_on (str): Left on
        right_on (str): Right on
        how (str, optional): How. Defaults to "inner".
        context (list[dict] | str, optional): Context. Defaults to None.
        model (str, optional): Model. Defaults to None.
        func (Callable, optional): User function. Defaults to None.
        strategy (Literal["nest", "block"], optional): Strategy. Defaults to "nest".
        limit (int): Maximum number of outputs to produce before stopping.
        rate_limit (int, optional): Rate limit. Defaults to 16.
        assertions (list[Callable], optional): Assertions. Defaults to [].
        batch_size (int, optional): Batch size for block join. Defaults to 5.
        **kwargs: Additional keyword arguments for OpenAI Clent.
    """
    
    join_op = JoinOperation(
        user_instruction=user_instruction,
        left_on=[left_on],
        right_on=[right_on],
        how=how,
        context=context,
        model=model,
        tool=FunctionCallTool(func=func) if func else None,
        strategy=strategy,
        limit=limit,
        rate_limit=rate_limit,
        assertions=assertions,
        batch_size=batch_size,
    )
    outputs = asyncio.run(join_op.execute(
        left_data=left_data,
        right_data=right_data,
        **kwargs
    ))
    return outputs


@dataclass
class JoinOpOutputs(BaseOpOutputs):
    join_pairs: list[tuple] = field(default_factory=list)
    left_join_keys: list[int] = field(default_factory=list)
    right_join_keys: list[int] = field(default_factory=list)


class JoinOperation(BaseOperation):
    """
    Join operator: Join values of two columns against a specific user's instruction.
    """
    strategy_options = ["nest", "block"]
    
    def __init__(
        self,
        user_instruction: str = "",
        left_on: list[str] = [],
        right_on: list[str] = [],
        how: str = "inner",
        context: list[dict] | str | None = None,
        model: str | None = None,
        tool: Callable | BaseTool | None = None,
        strategy: Literal["nest", "block"] = "nest",
        limit: int | None = None,
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
        batch_size: int = 5,
    ):
        if tool and not isinstance(tool, BaseTool):
            tool = FunctionCallTool.from_function(func=tool)
        
        super().__init__(
            op_name="join", 
            user_instruction=user_instruction,
            context=context,
            model=model,
            tool=tool,
            strategy=strategy,
            limit=limit,
            rate_limit=rate_limit,
            assertions=assertions,
        )
        self.prompter = JoinPrompter()
        self.left_on = left_on
        self.right_on = right_on
        self.how = how
        self.batch_size = batch_size

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
        if self.strategy == "block":
            kwargs["batch_size"] = self.batch_size
        return kwargs
    
    def _prepare_nested_join_pairs(self, left_values, right_values):
        left_ids = left_values.index
        right_ids = right_values.index
        data_id_pairs = [(left_id, right_id) for left_id in left_ids for right_id in right_ids]
        return data_id_pairs
    
    async def _execute_by_func(self, left_value: pd.Series, right_value: pd.Series, user_instruction: str, func: Callable, llm_call: Callable, **kwargs):
        try:
            result = func(left_value, right_value)
            return result, 0.0
        except Exception as e:
            warnings.warn(f"Evaluation by UDF failed with error {e}. Switch to LLM evaluation.")
            return await llm_call(left_value, right_value, user_instruction)

    async def _pairwise_evaluate(self, left_value: pd.Series, right_value: pd.Series, user_instruction: str, left_dtypes: list[str], right_dtypes: list[str], **kwargs):
        async with self.semaphore:
            full_prompt = self.prompter.generate_nested_join_prompt(left_value, right_value, user_instruction, left_dtypes, right_dtypes)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            result = self._process_nested_join_response(output.get("output", None))
            return result, output["cost"]
        
    def _process_nested_join_response(self, response: str | None) -> bool:
        if response is None:
            return False
        elif isinstance(response, bool):
            return response
        elif "true" in response.lower():
            return True
        elif "false" in response.lower():
            return False
        else:
            return False
    
    def _postprocess_nested_join_outputs(
        self, 
        data_id_pairs: list[tuple], 
        join_outputs: list[bool],
        how: str,
        left_keys: pd.Series,
        right_keys: pd.Series,
    ) -> tuple[list[tuple], list[int], list[int]]:
        key_mapping = {}
        join_pairs = []
        for can_join, (left_key, right_key) in zip(join_outputs, data_id_pairs):
            if can_join:
                join_pairs.append((left_key, right_key))
                # If left join, keys for right side are mapped to left side;
                # otherwise, keys for left side are mapped to right side.
                if how == "inner" or how == "left":
                    key_mapping[right_keys.loc[right_key]] = left_keys.loc[left_key]
                else:
                    key_mapping[left_keys.loc[left_key]] = right_keys.loc[right_key]
        
        if how == "inner" or how == "left":
            right_join_keys = right_keys.apply(lambda x: key_mapping.get(x, x)).to_list()
            left_join_keys = left_keys.to_list()
        else:
            left_join_keys = left_keys.apply(lambda x: key_mapping.get(x, x)).to_list()
            right_join_keys = right_keys.to_list()
        return join_pairs, left_join_keys, right_join_keys
    
    async def _nested_join(
        self, 
        left_data: pd.DataFrame, 
        right_data: pd.DataFrame, 
        user_instruction: str, 
        left_dtypes: list[str], 
        right_dtypes: list[str],
        **kwargs
    ):
        cache = kwargs.pop("cache", None)
        execution_func = functools.partial(self._pairwise_evaluate, left_dtypes=left_dtypes, right_dtypes=right_dtypes, model=self.model, **kwargs)
        # Prepare candidate pairs
        data_id_pairs = self._prepare_nested_join_pairs(left_data, right_data)
        left_keys = pd.Series(range(len(left_data)), index=left_data.index)
        right_keys = pd.Series(range(len(left_data), len(left_data) + len(right_data)), index=right_data.index)
        
        # Prepare values for join and their dtypes
        left_join_values: pd.DataFrame = left_data[self.left_on]
        right_join_values: pd.DataFrame = right_data[self.right_on]

        tasks = []
        for left_id, right_id in data_id_pairs:
            left_value: pd.Series = left_join_values.loc[left_id]
            right_value: pd.Series = right_join_values.loc[right_id]
            if left_value.empty or right_value.empty:
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=(False, 0.0))))
            elif cache is not None and (left_id, right_id) in cache:
                join_result = (cache[(left_id, right_id)], 0.0)
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=join_result)))
            elif self.has_udf():
                tasks.append(asyncio.create_task(self._execute_by_func(left_value, right_value, user_instruction, self.tool, execution_func)))
            else:
                tasks.append(asyncio.create_task(execution_func(left_value, right_value, user_instruction)))
        
        # Wait for all tasks to complete
        if self.limit is not None and self.limit <= 0:
            warnings.warn("The limit should be positive. To execute, the limit will be ignored.")
            self.limit = None
        
        token_cost = 0.0
        join_outputs: list[bool] = []
        if self.limit is not None:
            num_passed_pairs: int = 0
            reach_limit: bool = False
            for i in range(0, len(tasks), self.limit):
                if reach_limit:
                    break
                batch_tasks = tasks[i : i + self.limit]
                batch_results = await asyncio.gather(*batch_tasks)
                token_cost += sum(result[1] for result in batch_results)
                for can_join, _ in batch_results:
                    join_outputs.append(can_join)
                    if can_join:
                        num_passed_pairs += 1
                    if num_passed_pairs >= self.limit:
                        reach_limit = True
                        break
            num_remaining_pairs = len(data_id_pairs) - len(join_outputs)
            if num_remaining_pairs > 0:
                join_outputs.extend([False] * num_remaining_pairs)
        else:
            results = await asyncio.gather(*tasks)
            join_outputs = [result[0] for result in results]
            token_cost = sum(result[1] for result in results)

        joined_pairs, left_join_keys, right_join_keys = self._postprocess_nested_join_outputs(
            data_id_pairs, join_outputs, self.how, left_keys=left_keys, right_keys=right_keys
        )
        return JoinOpOutputs(
            join_pairs=joined_pairs,
            left_join_keys=left_join_keys,
            right_join_keys=right_join_keys,
            cost=token_cost,
        )
    
    def _prepare_join_batches(
        self, 
        left_values: pd.DataFrame,
        right_values: pd.DataFrame,
        batch_size: int,
    ) -> tuple[list[pd.DataFrame], list[list], list[pd.DataFrame], list[list]]:
        # Prepare left batches
        left_batches, left_keys = [], []
        start_idx = 0
        while start_idx < len(left_values):
            left_batches.append(left_values.iloc[start_idx : start_idx+batch_size].reset_index(drop=True))
            left_keys.append(left_values.index[start_idx : start_idx+batch_size].tolist())
            start_idx += batch_size
        
        # Prepare right batches
        right_batches, right_keys = [], []
        start_idx = 0
        while start_idx < len(right_values):
            right_batches.append(right_values.iloc[start_idx : start_idx+batch_size].reset_index(drop=True))
            right_keys.append(right_values.index[start_idx : start_idx+batch_size].tolist())
            start_idx += batch_size
        
        return left_batches, left_keys, right_batches, right_keys

    async def _batchwise_evaluate(
        self,
        left_batch: pd.DataFrame,
        right_batch: pd.DataFrame,
        user_instruction: str,
        left_dtypes: list[str],
        right_dtypes: list[str],
        keys_in_left_batch: list,
        keys_in_right_batch: list,
        **kwargs
    ):
        async with self.semaphore:
            full_prompt = self.prompter.generate_batch_join_prompt(left_batch, right_batch, user_instruction, left_dtypes, right_dtypes)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            result = self._process_batch_join_response(output.get("output", None), keys_in_left_batch, keys_in_right_batch)
            return result, output["cost"]
        
    def _process_batch_join_response(
        self,
        response: str | None,
        keys_in_left_batch: list,
        keys_in_right_batch: list,
    ) -> list[tuple]:
        if response is None or response == "":
            return []
        pairs_str = response.split(',')
        joined_pairs = []
        for pair_str in pairs_str:
            pair_str = pair_str.strip()
            left_ref, right_ref = pair_str.split('-')
            left_idx, right_idx = keys_in_left_batch[int(left_ref[1:])], keys_in_right_batch[int(right_ref[1:])]
            pair = (left_idx, right_idx)
            joined_pairs.append(pair)
        return joined_pairs
        
    def _postprocess_block_join_outputs(
        self,
        join_pairs: list[tuple],
        how: str,
        left_keys: pd.Series,
        right_keys: pd.Series,
    ):
        key_mapping = {}
        for left_key, right_key in join_pairs:
            # If left join, keys for right side are mapped to left side;
            # otherwise, keys for left side are mapped to right side.
            if how == "inner" or how == "left":
                key_mapping[right_keys.loc[right_key]] = left_keys.loc[left_key]
            else:
                key_mapping[left_keys.loc[left_key]] = right_keys.loc[right_key]

        if how == "inner" or how == "left":
            right_join_keys = right_keys.apply(lambda x: key_mapping.get(x, x)).to_list()
            left_join_keys = left_keys.to_list()
        else:
            left_join_keys = left_keys.apply(lambda x: key_mapping.get(x, x)).to_list()
            right_join_keys = right_keys.to_list()
        return join_pairs, left_join_keys, right_join_keys
    
    async def _block_join(
        self,
        left_data: pd.DataFrame,
        right_data: pd.DataFrame,
        user_instruction: str,
        batch_size: int,
        left_dtypes: list[str],
        right_dtypes: list[str],
        **kwargs
    ):
        cache = kwargs.pop("cache", None)
        execution_func = functools.partial(self._batchwise_evaluate, left_dtypes=left_dtypes, right_dtypes=right_dtypes, model=self.model, **kwargs)

        # Prepare batches
        left_join_values: pd.DataFrame = left_data[self.left_on]
        right_join_values: pd.DataFrame = right_data[self.right_on]
        left_batches, left_keys, right_batches, right_keys = self._prepare_join_batches(left_join_values, right_join_values, batch_size=batch_size)

        tasks, batch_ids_pairs = [], []
        for left_batch_id, (left_batch, keys_in_left_batch) in enumerate(zip(left_batches, left_keys)):
            for right_batch_id, (right_batch, keys_in_right_batch) in enumerate(zip(right_batches, right_keys)):
                batch_ids_pairs.append((left_batch_id, right_batch_id))
                if cache is not None and (left_batch_id, right_batch_id) in cache:
                    join_result = (cache[(left_batch_id, right_batch_id)], 0.0)
                    tasks.append(asyncio.create_task(asyncio.sleep(0, result=join_result)))
                else:
                    tasks.append(asyncio.create_task(execution_func(
                        left_batch, right_batch, user_instruction, keys_in_left_batch=keys_in_left_batch, keys_in_right_batch=keys_in_right_batch
                    )))
        
        # Wait for all tasks to complete
        if self.limit is not None and self.limit <= 0:
            warnings.warn("The limit should be positive. To execute, the limit will be ignored.")
            self.limit = None
        
        token_cost = 0.0
        join_outputs: list[tuple] = []
        if self.limit is not None:
            num_passed_pairs: int = 0
            reach_limit: bool = False
            for i in range(0, len(tasks), self.limit):
                if reach_limit:
                    break
                batch_tasks = tasks[i : i + self.limit]
                batch_results = await asyncio.gather(*batch_tasks)
                token_cost += sum(result[1] for result in batch_results)
                for join_pairs_in_batch, _ in batch_results:
                    margin = self.limit - num_passed_pairs
                    if len(join_pairs_in_batch) <= margin:
                        join_outputs.extend(join_pairs_in_batch)
                    else:
                        join_outputs.extend(join_pairs_in_batch[:margin])
                    num_passed_pairs += len(join_pairs_in_batch)
                    if num_passed_pairs >= self.limit:
                        reach_limit = True
                        break
        else:
            results = await asyncio.gather(*tasks)
            for result in results:
                join_outputs.extend(result[0])
            token_cost = sum(result[1] for result in results)

        left_join_keys = pd.Series(range(len(left_data)), index=left_data.index)
        right_join_keys = pd.Series(range(len(left_data), len(left_data) + len(right_data)), index=right_data.index)
        joined_pairs, left_join_keys, right_join_keys = self._postprocess_block_join_outputs(
            join_outputs, self.how, left_join_keys, right_join_keys
        )
        return JoinOpOutputs(
            join_pairs=joined_pairs,
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
        if self.user_instruction is None and not self.has_udf():
            raise ValueError("`user_instruction` or `tool` (e.g., a UDF) is required.")
        if left_data.empty or right_data.empty:
            return JoinOpOutputs(
                output=[],
                left_join_keys=[],
                right_join_keys=[],
                cost=0.0,
            )
        
        # Prepare dtypes for left and right join columns
        left_dtypes: list = []
        for col in self.left_on:
            if isinstance(left_data[col].dtype, ImageDtype):
                left_dtypes.append("image")
            else:
                left_dtypes.append("str")
        right_dtypes: list = []
        for col in self.right_on:
            if isinstance(right_data[col].dtype, ImageDtype):
                right_dtypes.append("image")
            else:
                right_dtypes.append("str")

        if self.strategy == "nest":
            return await self._nested_join(left_data, right_data, self.user_instruction, left_dtypes, right_dtypes, **kwargs)
        elif self.strategy == "block":
            if self.has_udf():
                warnings.warn("The block semantic join does not support user-defined functions for now.")
            return await self._block_join(left_data, right_data, self.user_instruction, self.batch_size, left_dtypes, right_dtypes, **kwargs)
        else:
            raise ValueError(f"The optional strategies available for join are {self.strategy_options}. Strategy {self.strategy} is not supported.")
