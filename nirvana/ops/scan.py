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
            num_samples: int | None = None,
    ):
        super().__init__(
            op_name="scan",
            user_instruction="",
        )
        self.source = source
        self.output_columns = output_columns
        self.num_samples = num_samples

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
    
    def set_sample_size(self, sample_size: int | None):
        if sample_size is not None:
            assert sample_size > 0, "Sample size must be positive."
        self.num_samples = sample_size
    
    async def scan_from_llm(self, *args, **kwargs):
        raise NotImplementedError("LLM scan operator is not implemented yet.")

    async def execute(
        self, 
        input_data: pd.DataFrame,
        **kwargs
    ):
        if self.source == "dataframe":
            if self.num_samples is not None:
                num_samples = min(self.num_samples, len(input_data))
                output_records = input_data.iloc[:num_samples]
            else:
                output_records = input_data
            return ScanOpOutputs(
                output=output_records,
                cost=0.0
            )
        elif self.source == "llm":
            raise NotImplementedError("LLM scan operator is not implemented yet.")
