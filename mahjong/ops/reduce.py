"""
Reduce: aggregate multiple data based on NL predicates
"""
from typing import Any, Iterable
from dataclasses import dataclass

from mahjong.ops.base import BaseOperation
from mahjong.prompt_templates.reduce_prompter import ReducePrompter


def reduce_helper(
        processed_data: Iterable[Any],
        user_instruction: str,
        **kwargs
):
    reduce_op = ReduceOperation()
    outputs = reduce_op.execute(
        processed_data=processed_data,
        user_instruction=user_instruction,
        **kwargs
    )
    return outputs


@dataclass
class ReduceOpOuputs:
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
            processed_data: Iterable[Any],
            user_instruction: str,
            *args,
            **kwargs
    ):
        full_prompt = self.prompter.generate_prompt(processed_data, user_instruction)
        outputs = self.llm(full_prompt, "output")
        return ReduceOpOuputs(**outputs)
