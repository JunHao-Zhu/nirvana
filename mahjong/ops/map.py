"""
Map: Perform a projecton on target data based on NL predicates
"""
import functools
from typing import Any, Iterable
from dataclasses import dataclass

from mahjong.ops.base import BaseOperation
from mahjong.prompt_templates.map_prompter import MapPrompter


def map_helper(
    processed_data: Iterable[Any], user_instruction: str, target_schema: str = None, strategy: str = None, **kwargs
):
    map_op = MapOperation()
    outputs = map_op.execute(
        processed_data=processed_data,
        user_instruction=user_instruction,
        target_schema=target_schema,
        strategy=strategy,
        **kwargs
    )
    return outputs


@dataclass
class MapOpOutputs:
    field_name: str = None
    output: Iterable[Any] = None


class MapOperation(BaseOperation):
    """ TODO: Implement MapOperation class """
    
    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__("map", *args, **kwargs)
        self.prompter = MapPrompter()
    
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

    def execute(
            self, 
            processed_data: Iterable[Any],
            user_instruction: str,
            target_schema: str = None,
            strategy: str = None,
            *args, 
            **kwargs
    ):
        """
        Executes a mapping operation on the provided data using the specified strategy.
        Args:
            processed_data (Iterable[Any]): The input data to be processed.
            user_instruction (str): The instruction provided by the user to guide the operation.
            target_schema (str, optional): The name of the field for the output. 
                This is required and must not be None.
            strategy (str, optional): The strategy to use for execution. Supported values are:
                - "plain_llm": Executes using a plain language model.
                - "cot": Executes using a chain-of-thought approach.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments. For the "cot" strategy, this may include:
                - demos: Examples or demonstrations to guide the chain-of-thought execution.
        Returns:
            MapOpOutputs: An object containing the field name and the processed output.
        Raises:
            ValueError: If `target_schema` is None.
            NotImplementedError: If the specified `strategy` is not supported.
        """
        if target_schema is None:
            raise ValueError("Field name for the output is required.")
        
        if strategy == "plain_llm":
            execution_func = functools.partial(self._plain_llm_execute)
        elif strategy == "cot":
            demos = kwargs.get("demos", None)
            execution_func = functools.partial(self._llm_cot_execute, demos=demos)
        else:
            raise NotImplementedError(f"Strategy {strategy} is not implemented.")
        
        outputs = execution_func(processed_data, user_instruction)
        return MapOpOutputs(
            field_name=target_schema,
            output=outputs
        )
