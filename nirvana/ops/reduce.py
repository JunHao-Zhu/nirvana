import asyncio
from typing import Any, Iterable, Callable
from dataclasses import dataclass, field
import pandas as pd
from pandas.api.types import is_numeric_dtype

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.executors.tools import FunctionCallTool
from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.reduce_prompter import ReducePrompter


def reduce_wrapper(
        input_data: Iterable[Any],
        user_instruction: str = None,
        input_column: str = None,
        func: Callable = None,
        **kwargs
):
    reduce_op = ReduceOperation(
        user_instruction=user_instruction,
        input_columns=[input_column],
        tool=FunctionCallTool.from_function(func=func) if func else None,
        **kwargs
    )
    outputs = asyncio.run(reduce_op.execute(
        input_data=input_data,
        **kwargs
    ))
    return outputs


@dataclass
class ReduceOpOutputs(BaseOpOutputs):
    output: Any = field(default=None)

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
            user_instruction: str = "",
            input_columns: list[str] = [],
            **kwargs,
    ):
        super().__init__(
            op_name="reduce",
            user_instruction=user_instruction,
            **kwargs
        )
        self.prompter = ReducePrompter()
        self.input_columns = input_columns

    @property
    def dependencies(self) -> list[str]:
        return self.input_columns
    
    @property
    def generated_fields(self) -> list[str]:
        return []
    
    @property
    def op_kwargs(self):
        kwargs = super().op_kwargs
        kwargs["input_columns"] = self.input_columns
        return kwargs

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
            *args,
            **kwargs
    ):
        if self.user_instruction is None and not self.has_udf():
            raise ValueError("Neither `user_instruction` nor `func` is given.")
        
        if input_data.empty:
            return ReduceOpOutputs(output="No data to process.")

        processed_data = input_data[self.input_columns[0]]
        if isinstance(processed_data.dtype, ImageDtype):
            dtype = "image"
        elif is_numeric_dtype(processed_data):
            dtype = "numeric"
        else:
            dtype = "str"

        reduce_results, token_cost = None, 0
        if self.has_udf() and dtype == "numeric":
            reduce_results, token_cost = await self._execute_by_func(processed_data, self.user_instruction, self.tool, self._execute_by_plain_llm, dtype, model=self.model, **kwargs)    
        else:
            reduce_results, token_cost = await self._execute_by_plain_llm(processed_data, self.user_instruction, dtype, model=self.model, **kwargs)

        return ReduceOpOutputs(
            output=reduce_results,
            cost=token_cost
        )
