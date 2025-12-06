import functools
import asyncio
from typing import Any, Iterable, Callable, Tuple
from dataclasses import dataclass, field
from numpy import amax
import pandas as pd

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.executors.tools import FunctionCallTool
from nirvana.ops.base import BaseOpOutputs, BaseOperation
from nirvana.ops.prompt_templates.map_prompter import MapPrompter


def map_wrapper(
    input_data: pd.DataFrame, 
    user_instruction: str = None,
    input_column: str = None,
    output_columns: list[str] = None, 
    func: Callable = None, 
    strategy: str = None,
    **kwargs
):
    map_op = MapOperation(
        user_instruction=user_instruction,
        implementation=strategy,
        input_columns=[input_column],
        output_columns=output_columns,
        tool=FunctionCallTool.from_function(func=func) if func else None,
        **kwargs
    )
    outputs = asyncio.run(map_op.execute(
        input_data=input_data,
        **kwargs
    ))
    return outputs


@dataclass
class MapOpOutputs(BaseOpOutputs):
    field_name: list[str] | None = field(default=None)
    output: dict[str, Iterable] = field(default_factory=list)

    def __add__(self, other: "MapOpOutputs"):
        assert self.field_name == other.field_name, "Cannot merge MapOpOutputs with different field names."
        map_output = dict()
        for name in self.field_name:
            map_output[name] = self.output[name] + other.output[name]
        return MapOpOutputs(
            field_name=self.field_name,
            output=map_output,
            cost=self.cost + other.cost
        )


class MapOperation(BaseOperation):
    """
    Map operator: Applies an LLM to perform a transformation on a column, producing a new column as output
    """
    implementation_options = ["plain", "self-refine", "vote"]
    
    def __init__(
            self,
            user_instruction: str = "",
            input_columns: list[str] = [],
            output_columns: list[str] = [],
            **kwargs,
    ):
        super().__init__(
            op_name="map",
            user_instruction=user_instruction,
            **kwargs
        )
        self.prompter = MapPrompter()
        self.input_columns = input_columns
        self.output_columns = output_columns

    @property
    def dependencies(self) -> list[str]:
        return self.input_columns
    
    @property
    def generated_fields(self) -> list[str]:
        return self.output_columns
    
    @property
    def op_kwargs(self):
        kwargs = super().op_kwargs
        kwargs["input_columns"] = self.input_columns
        kwargs["output_columns"] = self.output_columns
        return kwargs

    async def _execute_by_plain_llm(self, data: Any, user_instruction: str, dtype: str, **kwargs):
        async with self.semaphore:
            if dtype == "str":
                data = f"{kwargs['field_name']}: {str(data)}"
            full_prompt = self.prompter.generate_prompt(data, user_instruction, self.output_columns, dtype)
            output = await self.llm(full_prompt, parse_tags=True, tags=self.output_columns, **kwargs)
            return output

    async def _execute_by_fewshot_llm(self, data: Any, user_instruction: str, dtype: str, demos, **kwargs):
        async with self.semaphore:
            if dtype == "str":
                data = f"{kwargs['field_name']}: {str(data)}"
            full_prompt = self.prompter.generate_fewshot_prompt(data, user_instruction, self.output_columns, dtype, demos)
            output = await self.llm(full_prompt, parse_tags=True, tags=self.output_columns, **kwargs)
            return output
        
    async def _execute_by_self_refine(self, data: Any, user_instruction: str, dtype: str, **kwargs):
        async with self.semaphore:
            self_refine_cost = 0.0
            if dtype == "str":
                data = f"{kwargs['field_name']}: {str(data)}"
            generate_prompt = self.prompter.generate_prompt(data, user_instruction, self.output_columns, dtype)
            output = await self.llm(generate_prompt, parse_tags=True, tags=self.output_columns, **kwargs)
            self_refine_cost += output["cost"]

            evaluate_prompt = self.prompter.generate_evaluate_prompt(data, output["raw_output"], user_instruction, dtype)
            evaluate_output = await self.llm(evaluate_prompt, parse_tags=True, tags=["evaluation", "feedback"], **kwargs)
            self_refine_cost += evaluate_output["cost"]
            
            if "pass" in evaluate_output["evaluation"].lower():
                output["cost"] = self_refine_cost
                return output
            else:
                refine_prompt = self.prompter.generate_refine_prompt(data, output["raw_output"], user_instruction, self.output_columns, evaluate_output["feedback"], dtype)
                refine_output = await self.llm(refine_prompt, parse_tags=True, tags=self.output_columns, **kwargs)
                self_refine_cost += refine_output["cost"]
                refine_output["cost"] = self_refine_cost
                return refine_output

    async def _execute_by_func(self, data: Any, user_instruction: str, func: Callable, llm_call: Callable, **kwargs):
        try:
            if len(self.output_columns) > 1:
                raise NotImplementedError(
                    "For now, the function tool is allowed to process one-to-one mapping. ",
                    "How to accommodate one-to-many map function with any output needs addressed."
                )
            map_result = func(data)
            output = {self.output_columns[0]: map_result, "cost": 0.0}
            return output
        except Exception as e:
            return await llm_call(data, user_instruction)
    
    def _postprocess_map_outputs(self, results: Iterable[dict], output_columns: list[str]):
        outputs = {column: [] for column in output_columns}
        total_cost = 0.0
        for llm_response in results:
            if llm_response is None:
                for column in output_columns:
                    outputs[column].append(None)
                continue

            for column in output_columns:
                outputs[column].append(llm_response.get(column, None))
            total_cost += llm_response.get("cost", 0.0)
        return outputs, total_cost

    async def execute(
            self, 
            input_data: pd.DataFrame,
            *args, 
            **kwargs
    ):  
        if self.user_instruction is None and not self.has_udf():
            raise ValueError("Neither `user_instruction` nor `func` is given.")
        
        if input_data.empty:
            return MapOpOutputs(field_name=self.output_columns, output=[])

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
                tasks.append(asyncio.create_task(asyncio.sleep(0, result=None)))
            elif self.has_udf():
                tasks.append(asyncio.create_task(self._execute_by_func(data, self.user_instruction, self.tool, execution_func)))
            else:
                tasks.append(asyncio.create_task(execution_func(data, self.user_instruction)))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Process results
        map_results, token_cost = self._postprocess_map_outputs(results, self.output_columns)
        return MapOpOutputs(
            field_name=self.output_columns,
            output=map_results,
            cost=token_cost
        )
