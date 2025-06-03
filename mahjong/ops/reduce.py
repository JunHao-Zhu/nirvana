"""
Reduce: aggregate multiple data based on NL predicates
"""
from typing import Any, Iterable, Callable
from dataclasses import dataclass
import pandas as pd

from mahjong.dataframe.arrays.image import ImageDtype
from mahjong.ops.base import BaseOpOutputs, BaseOperation
from mahjong.ops.prompt_templates.reduce_prompter import ReducePrompter


def reduce_wrapper(
        input_data: Iterable[Any],
        user_instruction: str = None,
        func: Callable = None,
        input_column: str = None,
        **kwargs
):
    reduce_op = ReduceOperation()
    outputs = reduce_op.execute(
        input_data=input_data,
        user_instruction=user_instruction,
        func=func,
        input_column=input_column,
        **kwargs
    )
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
    TODO: Simple implementation that does not consider the case where the input length exceeds the token limit.
    The next step is to implement several optimizations, like `summarize and aggregate` and `incremental aggregation`
    """

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("reduce", *args, **kwargs)
        self.prompter = ReducePrompter()

    def _plain_llm_execute(self, processed_data: Iterable[Any], user_instruction: str, dtype: str, **kwargs):
        full_prompt = self.prompter.generate_prompt(processed_data, user_instruction, dtype)
        output = self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
        return output["output"], output["cost"]

    def execute(
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
        else:
            dtype = "str"

        reduce_results, token_cost = None, 0
        if func is not None:
            try:
                reduce_results = processed_data.agg(func)
                token_cost = 0
            except:
                reduce_results, token_cost = self._plain_llm_execute(processed_data, user_instruction, dtype, **kwargs)
        else:
            reduce_results, token_cost = self._plain_llm_execute(processed_data, user_instruction, dtype, **kwargs)

        return ReduceOpOutputs(
            output=reduce_results,
            cost=token_cost
        )
