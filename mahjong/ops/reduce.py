"""
Reduce: aggregate multiple data based on NL predicates
"""
from typing import Any, Iterable
from dataclasses import dataclass
import pandas as pd

from mahjong.ops.base import BaseOperation
from mahjong.prompt_templates.reduce_prompter import ReducePrompter


def reduce_wrapper(
        processed_data: Iterable[Any],
        user_instruction: str,
        input_column: str,
        **kwargs
):
    reduce_op = ReduceOperation()
    outputs = reduce_op.execute(
        processed_data=processed_data,
        user_instruction=user_instruction,
        input_column=input_column,
        **kwargs
    )
    return outputs


@dataclass
class ReduceOpOutputs:
    output: Any = None


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

    def execute(
            self,
            input_data: pd.DataFrame,
            user_instruction: str,
            input_column: str,
            *args,
            **kwargs
    ):
        processed_data = input_data[input_column]
        full_prompt = self.prompter.generate_prompt(processed_data, user_instruction)
        outputs = self.llm(full_prompt, "output")
        return ReduceOpOutputs(**outputs)
