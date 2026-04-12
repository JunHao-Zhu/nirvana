import warnings
from typing import Optional
from pydantic import BaseModel, Field, model_validator
from typing_extensions import deprecated
import pandas as pd

from nirvana.executors.llm_backbone import LLMClient
from nirvana.lineage.abstractions import LineageNode
from nirvana.optim.logical import LogicalOptimizer
from nirvana.optim.physical import PhysicalOptimizer
    

class OptimizeConfig(BaseModel):
    do_logical_optimization: bool = Field(default=True, description="whether perform logical plan optimization.")
    do_physical_optimization: bool = Field(default=True, description="whether perform physical plan optimization.")
    sample_size: Optional[int] = Field(default=5, description="The number of data used for plan optimization.")
    max_rounds: int = Field(default=5, description="The maximum number of optimization rounds for logical optimization.")
    improve_margin: float = Field(default=0.2, description="The improvement margin for physical optimization.")

    # transformation rules
    filter_pullup: bool = Field(default=True, description="Whether use filter pullup.")
    filter_pushdown: bool = Field(default=True, description="Whether use filter pushdown.")
    map_pullup: bool = Field(default=True, description="Whether use map pullup.")
    non_llm_pushdown: bool = Field(default=True, description="Whether use non-llm pushdown.")
    non_llm_replace: bool = Field(default=True, description="Whether use non-llm replacement")
    operator_fusion: bool = Field(default=True, description="Whether use operator fusion.")

    # available backend models for query optimization
    available_models: list[str] = Field(default_factory=list, description="The available models for physical optimization.")

    # deprecated fields
    approx_mode: bool = Field(
        default=True,
        description="Whether use approximation for physical optimization.",
        deprecated=deprecated("`approx_mode` is deprecated and will be removed in future versions."),
    )
    sample_ratio: Optional[float] = Field(
        default=None,
        description="The ratio of data used for physical optimization.",
        deprecated=deprecated("`sample_ratio` is deprecated and will be removed in future versions. Please use `sample_size` instead."),
    )
    avaiable_models: list[str] = Field(
        default_factory=list,
        description="Deprecated misspelling of `available_models`; kept for backward compatibility.",
        deprecated=deprecated("`avaiable_models` is a typo and will be removed in future versions. Please use `available_models` instead."),
    )

    @model_validator(mode="after")
    def _sync_available_models_alias(self) -> "OptimizeConfig":
        if self.avaiable_models and not self.available_models:
            warnings.warn(
                "`avaiable_models` is a typo and will be removed in future versions. Please use `available_models` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.available_models = self.avaiable_models
        return self


class PlanOptimizer:
    client: LLMClient = None

    @classmethod
    def set_agent(cls, client: LLMClient):
        cls.client = client

    def __init__(self, config: OptimizeConfig | dict = None):
        if isinstance(config, dict):
            config = OptimizeConfig(**config)
        self.config = config if config is not None else OptimizeConfig()
        self.prepare_optimizers(self.config)

    def set_config(self, config: OptimizeConfig):
        self.config = config

    def prepare_optimizers(self, config: OptimizeConfig):
        if self.config.sample_ratio:
            warnings.warn(
                "`sample_ratio` is deprecated and will be removed in future versions. Now we use default `sample_size` instead.",
                DeprecationWarning,
            )
        if config.do_logical_optimization:
            self.logical_optimizer = LogicalOptimizer(
                self.config.max_rounds, self.client, config.filter_pullup, config.filter_pushdown, config.map_pullup, config.non_llm_pushdown, config.non_llm_replace
            )
        else:
            self.logical_optimizer = None

        if config.do_physical_optimization:
            self.physical_optimizer = PhysicalOptimizer(self.client, config.available_models, config.sample_size, config.improve_margin)
        else:
            self.physical_optimizer = None

    def clear(self):
        if self.logical_optimizer:
            self.logical_optimizer.clear()

    def optimize_logical_plan(self, plan: LineageNode):
        plan = self.logical_optimizer.optimize(plan, num_samples=self.config.sample_size)
        return plan
    
    def optimize_physical_plan(self, plan: LineageNode):
        return self.physical_optimizer.optimize(plan)
