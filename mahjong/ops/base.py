from abc import ABC, abstractmethod


class BaseOperation(ABC):

    def __init__(self, op_name: str, *args, **kwargs):
        self.op_name = op_name

    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError(f"The operator {self.op_name} is not implemented.")
