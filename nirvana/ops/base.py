import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable
from abc import ABC, abstractmethod

from nirvana.executors.llm_backbone import LLMClient
from nirvana.executors.tools import BaseTool


@dataclass
class BaseOpOutputs:
    output: Any | None = field(default=None)
    cost: float = field(default=0.0)


class BaseOperation(ABC):
    llm: LLMClient = None

    def __init__(
        self, 
        op_name: str,
        user_instruction: str = "",
        context: list[dict] | str | None = None,
        model: str | None = None,
        tool: BaseTool | None = None,
        strategy: str | None = "plain",
        rate_limit: int = 16,
        assertions: list[Callable] | None = [],
    ):
        self.op_name = op_name
        self.user_instruction = user_instruction
        self.context = context
        self.model = model if model else self.llm.default_model
        self.tool = tool
        self.strategy = "plain" if strategy is None else strategy
        self.semaphore = asyncio.Semaphore(rate_limit)
        self.assertions = assertions

    @classmethod
    def set_llm(cls, llm: LLMClient):
        cls.llm = llm
        
    @property
    def op_kwargs(self) -> dict:
        return {
            "user_instruction": self.user_instruction,
            "context": self.context,
            "model": self.model,
            "tool": self.tool.__repr__(),
            "strategy": self.strategy,
            "assertions": self.assertions,
            "rate_limit": self.semaphore._value
        }

    @property
    def dependencies(self) -> list[str]:
        raise NotImplementedError("Subclasses must implement dependencies property.")

    @property
    def generated_fields(self) -> list[str]:
        raise NotImplementedError("Subclasses must implement generated_fields property.")

    def has_udf(self) -> bool:
        return callable(self.tool)

    def add_assertion(self, assertion: Callable | list[Callable]):
        if isinstance(assertion, list):
            self.assertions.extend(assertion)
        else:
            self.assertions.append(assertion)

    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError(f"The operator {self.op_name} is not implemented.")
