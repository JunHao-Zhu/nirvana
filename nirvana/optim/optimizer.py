from typing import List, Optional
from pydantic import BaseModel, Field
import pandas as pd

from nirvana.executors.llm_backbone import LLMClient
from nirvana.lineage.abstractions import LineageNode
from nirvana.optim.logical import LogicalOptimizer
from nirvana.optim.physical import PhysicalOptimizer
    

class OptimizeConfig(BaseModel):
    do_logical_optimization: bool = Field(default=True, description="whether perform logical plan optimization.")
    do_physical_optimization: bool = Field(default=True, description="whether perform physical plan optimization.")
    sample_ratio: Optional[float] = Field(default=None, description="The ratio of data used for physical optimization.")
    sample_size: Optional[int] = Field(default=None, description="The number of data used for physical optimization.")
    improve_margin: float = Field(default=0.2, description="The improvement margin for physical optimization.")
    approx_mode: bool = Field(default=True, description="Whether use approximation for physical optimization.")

    # transformation rules
    filter_pullup: bool = Field(default=True, description="Whether use filter pullup.")
    filter_pushdown: bool = Field(default=True, description="Whether use filter pushdown.")
    map_pullup: bool = Field(default=True, description="Whether use map pullup.")
    non_llm_pushdown: bool = Field(default=True, description="Whether use non-llm pushdown.")
    non_llm_replace: bool = Field(default=True, description="Whether use non-llm replacement")

    # available backend models for query optimization
    avaiable_models: list[str] = Field(default_factory=list, description="The available models for physical optimization.")


class PlanOptimizer:
    client: LLMClient = None

    @classmethod
    def set_agent(cls, client: LLMClient):
        cls.client = client

    def __init__(self, config: OptimizeConfig = None):
        self.config = config if config is not None else OptimizeConfig()
        if self.config.do_logical_optimization:
            self.logical_optimizer = LogicalOptimizer(self.client, config.non_llm_replace)
        else:
            self.logical_optimizer = None
        if self.config.do_physical_optimization:
            self.physical_optimizer = PhysicalOptimizer(self.client, config.avaiable_models)
        else:
            self.physical_optimizer = None

    def set_config(self, config: OptimizeConfig):
        self.config = config

    def clear(self):
        if self.logical_optimizer:
            self.logical_optimizer.clear()

    def optimize_logical_plan(self, plan: LineageNode, input_dataset_name: str, columns: List[str]):
        # plan = self.logical_optimizer.optimize(plan, input_dataset_name, columns)
        plan = self.logical_optimizer.optimize(plan)
        return plan
    
    def optimize_physical_plan(self, plan: LineageNode, input_data: pd.DataFrame):
        if self.config.sample_ratio:
            num_sample = int(self.config.sample_ratio * len(input_data)) + 1
        elif self.config.sample_size:
            num_sample = self.config.sample_size
        else:
            raise ValueError("Please specify either `sample_ratio` or `sample_size` for physical plan optimization.")
        return self.physical_optimizer.optimize(plan, input_data, num_sample, self.config.improve_margin, self.config.approx_mode)
