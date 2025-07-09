import functools
import asyncio
from typing import Any, Iterable, Callable
from dataclasses import dataclass
import pandas as pd

from mahjong.dataframe.arrays.image import ImageDtype
from mahjong.ops.base import BaseOpOutputs, BaseOperation
from mahjong.ops.prompt_templates.rank_prompter import RankPrompter


def rank_wrapper(**kwargs):
    pass


@dataclass
class RankOpOutputs(BaseOpOutputs):
    output: Iterable[int] = None

    def __add__(self, other: "RankOpOutputs"):
        return RankOpOutputs(
            output=self.output + other.output,
            cost=self.cost + other.cost
        )


class RankOperation(BaseOperation):
    """
    Rank operator: Ranks values in a column according to an NL-specified ranking function.

    TODO: test the rank operator
    """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("rank", *args, **kwargs)
        self.prompter = RankPrompter()

    async def _compare_by_plain_llm(self, data1: Any, data2: Any, user_instruction: str, dtype: str, **kwargs):
        async with self.semaphore:
            full_prompt = self.prompter.generate_prompt(data1, data2, user_instruction, dtype)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return output["output"], output["cost"]
        
    async def partition(self, data: Iterable, ranking: list, low: int, high: int, user_instruction: str, dtype: str):
        # Choose a pivot element
        pivot_index = ranking[(low + high) // 2]
        pivot_value = data[pivot_index]
        i = low - 1
        j = high + 1

        cost_for_partiton = 0.0
        while True:
            # Find an element on the left that is larger than the pivot
            while True:
                i += 1
                index_in_data = ranking[i]
                comparison_result, cost_per_comparison = await self._compare_by_plain_llm(data[index_in_data], pivot_value, user_instruction, dtype)
                cost_for_partiton += cost_per_comparison
                if int(comparison_result) == 1:  # if data[i] is larger than pivot_value, break
                    break

            # Find an element on the right that is smaller than the pivot
            while True:
                j -= 1
                index_in_data = ranking[j]
                comparison_result, cost_per_comparison = await self._compare_by_plain_llm(data[index_in_data], pivot_value, user_instruction, dtype)
                cost_for_partiton += cost_per_comparison
                if int(comparison_result) == 2:  # if data[j] is smaller than pivot_value, break
                    break

            # If i and j have crossed, break
            if i >= j:
                break

            # Swap data[i] and data[j]
            ranking[i], ranking[j] = ranking[j], ranking[i]

        # Swap the pivot element to its correct position
        ranking[i], ranking[pivot_index] = ranking[pivot_index], ranking[i]

        return i, cost_for_partiton
    
    async def quick_sort(self, data: Iterable, ranking: list, low: int, high: int, user_instruction: str, dtype: str, **kwargs):
        total_cost = 0.0
        if low < high:
            pivot_index, cost_for_partition = await self.partition(data, ranking, low, high, user_instruction, dtype)
            total_cost += cost_for_partition
            await self.quick_sort(data, ranking, low, pivot_index - 1, user_instruction, dtype, **kwargs)
            await self.quick_sort(data, ranking, pivot_index + 1, high, user_instruction, dtype, **kwargs)
        return ranking, total_cost

    async def execute(
            self,
            input_data: pd.DataFrame,
            user_instruction: str = None,
            func: Callable = None,
            input_column: str = None,
            *args,
            **kwargs,
    ):
        if user_instruction is None and func is None:
            raise ValueError("Neither `user_instruction` nor `func` is given.")
        
        if input_data.empty:
            return RankOpOutputs(output=None)

        processed_data = input_data[input_column]
        if isinstance(processed_data.dtype, ImageDtype):
            dtype = "image"
        else:
            dtype = "str"

        low, high = 0, len(processed_data) - 1
        ranking = list(range(len(processed_data)))
        ranking, token_cost = await self.quick_sort(processed_data, ranking, low, high, user_instruction, dtype, **kwargs)

        return RankOpOutputs(
            output=ranking,
            cost=token_cost
        )
