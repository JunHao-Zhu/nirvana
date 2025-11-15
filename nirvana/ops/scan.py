import functools
import asyncio
from typing import Any, Literal
from dataclasses import dataclass, field
import pandas as pd

from nirvana.dataframe.arrays.image import ImageDtype
from nirvana.ops.base import BaseOpOutputs, BaseOperation


@dataclass
class ScanOpOutputs(BaseOpOutputs):
    output: Any = field(default=None)


class ScanOperation(BaseOperation):
    """
    Scan operator: Extract data from a data source or an LLM
    """
    
    def __init__(
            self,
            source: Literal["dataframe", "llm"] = "dataframe",
            output_columns: list[str] = [],
            **kwargs,
    ):
        super().__init__(
            op_name="scan",
            user_instruction="",
            **kwargs
        )
        self.source = source
        self.output_columns = output_columns

    @property
    def dependencies(self) -> list[str]:
        return []
    
    @property
    def generated_fields(self) -> list[str]:
        return self.output_columns
    
    @property
    def op_kwargs(self):
        kwargs = super().op_kwargs
        kwargs["source"] = self.source
        kwargs["output_columns"] = self.output_columns
        return kwargs

    async def execute(
            self, 
            input_data: pd.DataFrame,
            *args, 
            **kwargs
    ):
        if self.source == "dataframe":
            return ScanOpOutputs(
                output=input_data,
                cost=0.0
            )
        elif self.source == "llm":
            raise NotImplementedError("LLM scan operator is not implemented yet.")
