from abc import ABC, abstractmethod

from mahjong.models.llm_backbone import LLMClient


class BaseOperation(ABC):
    llm: LLMClient = None

    def __init__(self, op_name: str, *args, **kwargs):
        self.op_name = op_name

    def set_llm(self, llm: LLMClient):
        self.llm = llm

    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError(f"The operator {self.op_name} is not implemented.")
