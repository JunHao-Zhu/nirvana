from typing import List, Optional
from dataclasses import dataclass, field
import json
import pandas as pd

from nirvana.models.llm_backbone import LLMClient
from nirvana.lineage.abstractions import LineageNode
from nirvana.optim.logical import LogicalOptimizer
from nirvana.optim.physical import PhysicalOptimizer


@dataclass
class OptimizeConfig:
    do_logical_optimization: bool = field(default=True, metadata={"help": "whether perform logical plan optimize."})
    do_physical_optimization: bool = field(default=True, metadata={"help": "whether perform physical plan optimize."})
    max_round: int = field(default=3, metadata={"help": "The maximum number of rounds calling agentic logical optimization."})
    sample_ratio: Optional[float] = field(default=None, metadata={"help": "The ratio of data used for physical optimization."})
    sample_size: Optional[int] = field(default=None, metadata={"help": "The number of data used for physical optimization."})
    improve_margin: float = field(default=0.2, metadata={"help": "The margin of improvement for physical optimization."})
    approx_mode: bool = field(default=True, metadata={"help": "Whether use approximation for physical optimization."})

    def to_json(self):
        return {
            "do_logical_optimization": self.do_logical_optimization,
            "do_physical_optimization": self.do_physical_optimization,
            "max_round": self.max_round,
            "sample_ratio": self.sample_ratio,
            "sample_size": self.sample_size,
            "improve_margin": self.improve_margin,
            "approx_mode": self.approx_mode
        }
    
    def __str__(self):
        config_json = self.to_json()
        return json.dumps(config_json, indent=2)


class PlanOptimizer:
    client: LLMClient = None

    @classmethod
    def set_agent(cls, client: LLMClient):
        cls.client = client

    def __init__(self, config: OptimizeConfig = None):
        self.config = config if config is not None else OptimizeConfig()
        if self.config.do_logical_optimization:
            self.logical_optimizer = LogicalOptimizer(self.config.max_round, self.client)
        else:
            self.logical_optimizer = None
        if self.config.do_physical_optimization:
            self.physical_optimizer = PhysicalOptimizer(self.client)
        else:
            self.physical_optimizer = None

    def set_config(self, config: OptimizeConfig):
        self.config = config

    def clear(self):
        if self.logical_optimizer:
            self.logical_optimizer.clear()

    def optimize_logical_plan(self, plan: LineageNode, input_dataset_name: str, columns: List[str]):
        plan = self.logical_optimizer.optimize(plan, input_dataset_name, columns)
        return plan
    
    def optimize_physical_plan(self, plan: LineageNode, input_data: pd.DataFrame):
        if self.config.sample_ratio:
            num_sample = int(self.config.sample_ratio * len(input_data)) + 1
        elif self.config.sample_size:
            num_sample = self.config.sample_size
        else:
            raise ValueError("Please specify either `sample_ratio` or `sample_size` for physical plan optimization.")
        return self.physical_optimizer.optimize(plan, input_data, num_sample, self.config.improve_margin, self.config.approx_mode)
