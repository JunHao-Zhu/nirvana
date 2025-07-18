import asyncio
from typing import Any, Iterable, Callable
from dataclasses import dataclass
import pandas as pd
from pandas.api.types import is_numeric_dtype

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.reduce_prompter import ReducePrompter


def reduce_wrapper(
        input_data: Iterable[Any],
        user_instruction: str = None,
        func: Callable = None,
        input_column: str = None,
        **kwargs
):
    reduce_op = ReduceOperation()
    outputs = asyncio.run(reduce_op.execute(
        input_data=input_data,
        user_instruction=user_instruction,
        func=func,
        input_column=input_column,
        **kwargs
    ))
    return outputs


@dataclass
class ReduceOpOutputs(BaseOpOutputs):
    output: Any = None

    def __add__(self, other: "ReduceOpOutputs"):
        return ReduceOpOutputs(
            output=self.output + other.output,
            cost=self.cost + other.cost
        )


class ReduceOperation(BaseOperation):
    """
    Reduce operator: Aggregates values in a column based on an NL-specified reduction function

    NOTE: This is a simple implementation that does not consider the case where the input length exceeds the token limit. 
          The next step is to implement several optimizations, like `summarize and aggregate` and `incremental aggregation`
    """

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("reduce", *args, **kwargs)
        self.prompter = ReducePrompter()
        rate_limit = kwargs.get("rate_limit", 16)
        self.semaphore = asyncio.Semaphore(rate_limit)

    async def _execute_by_plain_llm(self, processed_data: Iterable[Any], user_instruction: str, dtype: str, **kwargs):
        async with self.semaphore:
            full_prompt = self.prompter.generate_prompt(processed_data, user_instruction, dtype)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return output["output"], output["cost"]
    
    async def _execute_by_func(self, processed_data: Iterable[Any], user_instruction: str, func: Callable, llm_call: Callable, dtype: str, **kwargs):
        try:
            reduce_results = processed_data.agg(func)
            return reduce_results, 0
        except:
            return await llm_call(processed_data, user_instruction, dtype, **kwargs)

    async def execute(
            self,
            input_data: pd.DataFrame,
            user_instruction: str = None,
            func: Callable = None,
            input_column: str = None,
            *args,
            **kwargs
    ):
        if user_instruction is None and func is None:
            raise ValueError("Neither `user_instruction` nor `func` is given.")
        
        if input_data.empty:
            return ReduceOpOutputs(output="No data to process.")

        processed_data = input_data[input_column]
        if isinstance(processed_data.dtype, ImageDtype):
            dtype = "image"
        elif is_numeric_dtype(processed_data):
            dtype = "numeric"
        else:
            dtype = "str"

        reduce_results, token_cost = None, 0
        if func is not None and dtype == "numeric":
            reduce_results, token_cost = await self._execute_by_func(processed_data, user_instruction, func, self._execute_by_plain_llm, dtype, **kwargs)    
        else:
            reduce_results, token_cost = await self._execute_by_plain_llm(processed_data, user_instruction, dtype, **kwargs)

        return ReduceOpOutputs(
            output=reduce_results,
            cost=token_cost
        )
