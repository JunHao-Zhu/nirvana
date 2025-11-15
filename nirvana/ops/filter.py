import functools
import asyncio
from typing import Any, Union, Iterable, Callable, Tuple
from dataclasses import dataclass
import pandas as pd

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.executors.tools import FunctionCallTool
from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.filter_prompter import FilterPrompter


def filter_wrapper(
    input_data: pd.DataFrame, 
    user_instruction: str = None,
    input_column: str = None,
    func: Callable = None,
    strategy: str = None,
    **kwargs
):
    filter_op = FilterOperation(
        user_instruction=user_instruction,
        input_columns=[input_column],
        tool=FunctionCallTool.from_function(func=func) if func else None,
        implementation=strategy,
        **kwargs
    )
    outputs = asyncio.run(filter_op.execute(
        input_data=input_data,
        **kwargs
    ))
    return outputs


@dataclass
class FilterOpOutputs(BaseOpOutputs):
    output: Iterable[bool] = None

    def __add__(self, other: "FilterOpOutputs"):
        return FilterOpOutputs(
            output=self.output + other.output,
            cost=self.cost + other.cost
        )


class FilterOperation(BaseOperation):
    """
    Filter operator: Uses an LLM to evaluate a natural language predicate on a column
    """
    implementation_options = ["plain", "self-refine", "vote"]
    
    def __init__(
            self,
            user_instruction: str = "",
            input_columns: list[str] = [],
            **kwargs,
    ):
        super().__init__(
            op_name="filter",
            user_instruction=user_instruction,
            **kwargs
        )
        self.prompter = FilterPrompter()
        self.input_columns = input_columns

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
        return kwargs

    async def _execute_by_plain_llm(self, data: Any, user_instruction: str, dtype: str, **kwargs):
        async with self.semaphore:
            if dtype == "str":
                data = f"{kwargs['field_name']}: {str(data)}"
            full_prompt = self.prompter.generate_prompt(data, user_instruction, dtype)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return output["output"], output["cost"]

    async def _execute_by_fewshot_llm(self, data: Any, user_instruction: str, dtype: str, demos, **kwargs):
        async with self.semaphore:
            if dtype == "str":
                data = f"{kwargs['field_name']}: {str(data)}"
            full_prompt = self.prompter.generate_fewshot_prompt(data, user_instruction, dtype, demos)
            output = await self.llm(full_prompt, parse_tags=True, tags=["output"], **kwargs)
            return output["output"], output["cost"]
        
    async def _execute_by_self_refine(self, data: Any, user_instruction: str, dtype: str, **kwargs):
        async with self.semaphore:
            self_refine_cost = 0.0
            if dtype == "str":
                data = f"{kwargs['field_name']}: {str(data)}"
            generate_prompt = self.prompter.generate_prompt(data, user_instruction, dtype)
            output = await self.llm(generate_prompt, parse_tags=True, tags=["output"], **kwargs)
            self_refine_cost += output["cost"]

            evaluate_prompt = self.prompter.generate_evaluate_prompt(data, output["raw_output"], user_instruction, dtype)
            evaluate_output = await self.llm(evaluate_prompt, parse_tags=True, tags=["evaluation", "feedback"], **kwargs)
            self_refine_cost += evaluate_output["cost"]
            
            if "pass" in evaluate_output["evaluation"].lower():
                return output["output"], self_refine_cost
            else:
                refine_prompt = self.prompter.generate_refine_prompt(data, output["raw_output"], user_instruction, evaluate_output["feedback"], dtype)
                refine_output = await self.llm(refine_prompt, parse_tags=True, tags=["output"], **kwargs)
                self_refine_cost += refine_output["cost"]
                return refine_output["output"], self_refine_cost
    
    async def _execute_by_func(self, data: Any, user_instruction: str, func: Callable, llm_call: Callable, **kwargs):
        try:
            output = func(data)
            return output, 0.0
        except Exception as e:
            return await llm_call(data, user_instruction)
    
    def _postprocess_filter_outputs(self, results: Iterable[Tuple[Any, float]]):
        outputs, costs = [], 0.0
        for output, cost in results:
            if output is None:
                outputs.append(False)
                continue
            if isinstance(output, bool):
                outputs.append(output)
                continue
            if "True" in output:
                outputs.append(True)
            elif "False" in output:
                outputs.append(False)
            costs += cost
        return outputs, costs

    async def execute(
            self, 
            input_data: pd.DataFrame,
            *args, 
            **kwargs
    ):
        if self.user_instruction is None and not self.has_udf():
            raise ValueError("Neither `user_instruction` nor `func` is given.")
        
        if input_data.empty:
            return FilterOpOutputs()
        
        processed_data = input_data[self.input_columns[0]]
        if isinstance(processed_data.dtype, ImageDtype):
            dtype = "image"
        else:
            dtype = "str"

        if self.implementation == "plain":
            execution_func = functools.partial(self._execute_by_plain_llm, dtype=dtype, field_name=self.input_columns[0], model=self.model, **kwargs)
        elif self.implementation == "fewshot":
            assert self.context is not None, "Few-shot examples must be provided in the context for in-context learning."
            demos = self.context
            execution_func = functools.partial(self._execute_by_fewshot_llm, dtype=dtype, demos=demos, field_name=self.input_columns[0], model=self.model, **kwargs)
        elif self.implementation == "self_refine":
            execution_func = functools.partial(self._execute_by_self_refine, dtype=dtype, field_name=self.input_columns[0], model=self.model, **kwargs)
        else:
            raise NotImplementedError(f"Strategy {self.implementation} is not implemented.")

        # Create tasks for all data points
        tasks = []
        for data in processed_data:
            if pd.isna(data):
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=(False, 0.0))))
            elif self.has_udf():
                tasks.append(asyncio.create_task(self._execute_by_func(data, self.user_instruction, self.tool, execution_func)))
            else:
                tasks.append(asyncio.create_task(execution_func(data, self.user_instruction)))

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Process results
        filter_outputs, token_cost = self._postprocess_filter_outputs(results)
        return FilterOpOutputs(
            output=filter_outputs,
            cost=token_cost
        )
