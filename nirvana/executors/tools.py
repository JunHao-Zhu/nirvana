from typing import Any, Callable, Awaitable
from abc import ABC, abstractmethod


class BaseTool(ABC):
    name: str
    description: str

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def __call__(self, *args, **kwds) -> Any:
        pass


class PythonInterpreterTool(BaseTool):
    descrption = "Python Interpreter that runs code in a sandboxed environment"

    def __call__(self, *args, **kwds):
        pass


class FunctionCallTool(BaseTool):
    description = "This is a tool that evaluates python code. It can be used to perform calculations."

    def __init__(
            self,
            name: str | None = None,
            func: Callable | None = None,
            coroutine: Callable[..., Awaitable[Any]] | None = None,
            **kwargs
    ):
        self.name = name
        self.func = func
        self.coroutine = coroutine

    def __repr__(self) -> str:
        return f"FunctionCall(name={self.name})"
    
    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_function(
        cls,
        func: Callable | None = None,
        coroutine: Callable[..., Awaitable[Any]] | None = None,
        name: str | None = None,
    ):
        if func is not None:
            func = func
        elif coroutine is not None:
            func = coroutine
        else:
            raise ValueError("Either `func` or `coroutine` must be provided.")
        
        name = name or func.__name__

        return cls(
            name=name,
            func=func,
            coroutine=coroutine
        )

    def _run(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
    async def _arun(self, *args, **kwargs):
        return await self.coroutine(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.func is not None:
            return self._run(*args, **kwargs)
        elif self.coroutine is not None:
            return self._arun(*args, **kwargs)
