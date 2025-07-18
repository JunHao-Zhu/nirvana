import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod

from nirvana.models.llm_backbone import LLMClient


@dataclass
class BaseOpOutputs:
    cost: float = 0.0


class BaseOperation(ABC):
    llm: LLMClient = None

    def __init__(self, op_name: str, *args, **kwargs):
        self.op_name = op_name

    @classmethod
    def set_llm(cls, llm: LLMClient):
        cls.llm = llm

    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError(f"The operator {self.op_name} is not implemented.")
