import warnings

import asyncio
from typing import Any, Callable, Literal
from dataclasses import dataclass, field
import pandas as pd

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.executors.tools import BaseTool, FunctionCallTool
from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.rank_prompter import RankPrompter


def rank_wrapper(
    input_data: pd.DataFrame,
    user_instruction: str = None,
    input_column: str = None,
    descend: bool = True,
    context: list[dict] | str | None = None,
    model: str | None = None,
    func: Callable = None,
    strategy: Literal["plain"] = "plain",
    rate_limit: int = 16,
    assertions: list[Callable] | None = [],
    **kwargs
):
    """
    A function wrapper for rank operation

    Args:
        input_data (pd.DataFrame): Input dataframe
        user_instruction (str, optional): User instruction. Defaults to None.
        input_column (str, optional): Input column. Defaults to None.
        descend (bool, optional): Whether to rank in descending order (True) or ascending order (False). Defaults to True.
        context (list[dict] | str, optional): Context. Defaults to None.
        model (str, optional): Model. Defaults to None.
        func (Callable, optional): User function. Defaults to None.
        strategy (Literal["plain"], optional): Strategy. Defaults to "plain".
        rate_limit (int, optional): Rate limit. Defaults to 16.
        assertions (list[Callable], optional): Assertions. Defaults to [].
        **kwargs: Additional keyword arguments for OpenAI Clent.
    """

    rank_op = RankOperation(
        user_instruction=user_instruction,
        input_columns=[input_column],
        descend=descend,
        context=context,
        model=model,
        tool=FunctionCallTool.from_function(func=func) if func else None,
        strategy=strategy,
        rate_limit=rate_limit,
        assertions=assertions
    )
    outputs = asyncio.run(rank_op.execute(
        input_data=input_data,
        **kwargs
    ))
    ranking = sorted(range(1, len(outputs.ranked_indices) + 1), key=lambda k: outputs.ranked_indices[k - 1])
    outputs.ranking = ranking
    return outputs


@dataclass
class RankOpOutputs(BaseOpOutputs):
    ranking: list[int] = field(default_factory=list)
    ranked_indices: list[int] = field(default_factory=list)


class RankOperation(BaseOperation):
    """
    RankOperation ranks the rows of a DataFrame column according to a user-specified natural language instruction.
    """
    
    def __init__(
        self,
        user_instruction: str = "",
        input_columns: list[str] = [],
        descend: bool = True,
        context: list[dict] | str | None = None,
        model: str | None = None,
        tool: BaseTool | None = None,
        strategy: Literal["plain"] = "plain",
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
    ):
        super().__init__(
            op_name="rank",
            user_instruction=user_instruction,
            context=context,
            model=model,
            tool=tool,
            strategy=strategy,  
            rate_limit=rate_limit,
            assertions=assertions,
        )
        self.prompter = RankPrompter()
        self.input_columns = input_columns
        self.descend = descend

    @property
    def dependencies(self) -> list[str]:
        return self.input_columns
    
    @property
    def generated_fields(self) -> list[str]:
        return []
    
    @property
    def op_kwargs(self) -> dict:
        kwargs = super().op_kwargs
        kwargs["input_columns"] = self.input_columns
        kwargs["descend"] = self.descend
        return kwargs

    async def _compare_by_plain_llm(self, data1: Any, data2: Any, user_instruction: str, dtype: str, **kwargs):
        async with self.semaphore:
            if dtype == "str":
                data1 = self.input_columns[0] + ": " + str(data1)
                data2 = self.input_columns[0] + ": " + str(data2)
            full_prompt = self.prompter.generate_prompt(data1, data2, user_instruction, dtype)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return output["output"], output["cost"]
        
    async def partition(self, data: pd.Series, ranking: list, low: int, high: int, user_instruction: str, dtype: str, **kwargs):
        cache = kwargs.get("cache", None)
        # Choose a pivot element
        pivot_index = ranking[(low + high) // 2]
        pivot_value = data.iloc[pivot_index]
        i = low - 1
        j = high + 1

        cost_for_partiton = 0.0
        while True:
            # Find an element on the left that is smaller than the pivot (less satisfy the instruction)
            while True:
                i += 1
                index_in_data = ranking[i]
                if index_in_data == pivot_index:
                    break
                if cache is not None and (index_in_data, pivot_index) in cache:
                    comparison_result = cache[(index_in_data, pivot_index)]
                    cost_per_comparison = 0.0
                else:
                    comparison_result, cost_per_comparison = await self._compare_by_plain_llm(data.iloc[index_in_data], pivot_value, user_instruction, dtype, **kwargs)
                cost_for_partiton += cost_per_comparison
                if int(comparison_result) == 2:  # if data[i] is smaller than pivot_value, break
                    break

            # Find an element on the right that is larger than the pivot (more satisfy the instruction)
            while True:
                j -= 1
                index_in_data = ranking[j]
                if index_in_data == pivot_index:
                    break
                if cache is not None and (index_in_data, pivot_index) in cache:
                    comparison_result = cache[(index_in_data, pivot_index)]
                    cost_per_comparison = 0.0
                else:
                    comparison_result, cost_per_comparison = await self._compare_by_plain_llm(data.iloc[index_in_data], pivot_value, user_instruction, dtype, **kwargs)
                cost_for_partiton += cost_per_comparison
                if int(comparison_result) == 1:  # if data[j] is larger than pivot_value, break
                    break

            # If i and j have crossed, break
            if i >= j:
                break
            # Swap data[i] and data[j]
            ranking[i], ranking[j] = ranking[j], ranking[i]

        return j, cost_for_partiton
    
    async def quick_sort(self, data: pd.Series, ranking: list, low: int, high: int, user_instruction: str, dtype: str, **kwargs):
        total_cost = 0.0
        if low < high:
            partition_position, cost_for_partition = await self.partition(data, ranking, low, high, user_instruction, dtype, **kwargs)
            total_cost += cost_for_partition
            ranking, cost_from_next_sort = await self.quick_sort(data, ranking, low, partition_position, user_instruction, dtype, **kwargs)
            total_cost += cost_from_next_sort
            ranking, cost_from_next_sort = await self.quick_sort(data, ranking, partition_position + 1, high, user_instruction, dtype, **kwargs)
            total_cost += cost_from_next_sort
        return ranking, total_cost

    async def execute(
        self,
        input_data: pd.DataFrame,
        **kwargs,
    ):
        if self.user_instruction is None and not self.has_udf():
            raise ValueError("Neither `user_instruction` nor `func` is given.")
        
        if input_data.empty:
            return RankOpOutputs(ranked_indices=[], cost=0.0)

        processed_data = input_data[self.input_columns[0]]
        if isinstance(processed_data.dtype, ImageDtype):
            dtype = "image"
        else:
            dtype = "str"

        if self.has_udf():
            warnings.warn("The udf is not supported in the current Rank operator implementation. Switch to LLM-based ranking.")

        low, high = 0, len(processed_data) - 1
        ranking = list(range(len(processed_data)))
        ranking, token_cost = await self.quick_sort(processed_data, ranking, low, high, self.user_instruction, dtype, model=self.model, **kwargs)
        if not self.descend:
            ranking.reverse()

        return RankOpOutputs(
            ranked_indices=ranking,
            cost=token_cost
        )
