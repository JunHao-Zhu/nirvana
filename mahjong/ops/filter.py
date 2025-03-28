"""
Filter: remove data that violates NL predicates.
"""
import functools
from typing import Any, Iterable
from dataclasses import dataclass

from mahjong.ops.base import BaseOperation
from mahjong.prompt_templates.filter_prompter import FilterPrompter


def filter_helper(
    processed_data: Iterable[Any], user_instruction: str, strategy: str = None, **kwargs
):
    filter_op = FilterOperation()
    outputs = filter_op.execute(
        processed_data=processed_data,
        user_instruction=user_instruction,
        strategy=strategy,
        **kwargs
    )
    return outputs


@dataclass
class FilterOpOutputs:
    output: Iterable[bool] = None


class FilterOperation(BaseOperation):
    """ TODO: Implement FilterOperation class """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("filter", *args, **kwargs)
        self.prompter = FilterPrompter()
    
    def _plain_llm_execute(self, processed_data: Iterable[Any], user_instruction: str):
        outputs = []
        for data in processed_data:
            full_prompt = self.prompter.generate_prompt(user_instruction, data)
            output = self.llm(full_prompt, "output")
            outputs.append(output["output"])
        return outputs

    def _llm_cot_execute(self, processed_data: Iterable[Any], user_instruction: str, demos):
        outputs = []
        for data in processed_data:
            full_prompt = self.prompter.generate_cot_prompt(user_instruction, data, demos)
            output = self.llm(full_prompt, "output")
            outputs.append(output["output"])
        return outputs
    
    def _postprocess_llm_outputs(llm_outputs: Iterable[str]):
        outputs = []
        for output in llm_outputs:
            if "True" in output:
                outputs.append(True)
            elif "False" in output:
                outputs.append(False)
            else:
                raise ValueError("The llm outputs do not contain True or False.")
        return outputs

    def execute(
            self, 
            processed_data: Iterable[Any],
            user_instruction: str,
            strategy: str = None,
            *args, 
            **kwargs
    ):
        if strategy == "plain_llm":
            execution_func = functools.partial(self._plain_llm_execute)
        elif strategy == "cot":
            demos = kwargs.get("demos", None)
            execution_func = functools.partial(self._llm_cot_execute, demos=demos)
        else:
            raise NotImplementedError(f"Strategy {strategy} is not implemented.")
        
        outputs = execution_func(processed_data, user_instruction)
        
        outputs = self._postprocess_llm_outputs(outputs)
        return FilterOpOutputs(
            output=outputs
        )
