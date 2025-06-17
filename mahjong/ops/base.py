import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod

from mahjong.models.llm_backbone import LLMClient


@dataclass
class BaseOpOutputs:
    cost: int = 0


class BaseOperation(ABC):
    llm: LLMClient = None

    def __init__(self, op_name: str, *args, **kwargs):
        max_concurrency = kwargs.get("max_concurrency", 64)
        self.op_name = op_name
        self.semaphore = asyncio.Semaphore(max_concurrency)  # Limit to 16 concurrent tasks

    @classmethod
    def set_llm(cls, llm: LLMClient):
        cls.llm = llm

    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError(f"The operator {self.op_name} is not implemented.")
