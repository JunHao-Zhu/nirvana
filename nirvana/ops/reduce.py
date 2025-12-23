import asyncio
from typing import Any, Iterable, Callable, Literal
from dataclasses import dataclass, field
import pandas as pd
from pandas.api.types import is_numeric_dtype

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.executors.tools import BaseTool, FunctionCallTool
from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.reduce_prompter import ReducePrompter


def reduce_wrapper(
    input_data: Iterable[Any],
    user_instruction: str = None,
    input_column: str = None,
    context: list[dict] | str | None = None,
    model: str | None = None,
    func: Callable = None,
    strategy: Literal["plain"] = "plain",
    rate_limit: int = 16,
    assertions: list[Callable] | None = [],
    **kwargs
):
    """
    A function wrapper for reduce operation

    Args:
        input_data (Iterable[Any]): Input data
        user_instruction (str, optional): User instruction. Defaults to None.
        input_column (str, optional): Input column. Defaults to None.
        context (list[dict] | str, optional): Context. Defaults to None.
        model (str, optional): Model. Defaults to None.
        func (Callable, optional): User function. Defaults to None.
        strategy (Literal["plain"], optional): Strategy. Defaults to "plain".
        rate_limit (int, optional): Rate limit. Defaults to 16.
        assertions (list[Callable], optional): Assertions. Defaults to [].
        **kwargs: Additional keyword arguments for OpenAI Clent.
    """
    
    reduce_op = ReduceOperation(
        user_instruction=user_instruction,
        input_columns=[input_column],
        context=context,
        model=model,
        tool=FunctionCallTool.from_function(func=func) if func else None,
        strategy=strategy,
        rate_limit=rate_limit,
        assertions=assertions,
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
        context: list[dict] | str | None = None,
        model: str | None = None,
        tool: BaseTool | None = None,
        strategy: Literal["plain"] = "plain",
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
    ):
        super().__init__(
            op_name="reduce",
            user_instruction=user_instruction,
            context=context,
            model=model,
            tool=tool,
            strategy=strategy,
            rate_limit=rate_limit,
            assertions=assertions,
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
