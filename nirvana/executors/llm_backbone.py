import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from nirvana.executors.constants import MODEL_PRICING, LLMProviders

logger = logging.getLogger(__name__)


def _get_api_key_from_file(file: Path) -> str:
    with open(file, 'r') as api_file:
        api_key = api_file.readline()
    return api_key


def _get_openai_compatible_provider_info(
    model_name: str,
    api_key: str | Path | None = None,
    base_url: str | None = None
) -> tuple[str, str]:
    """Infer provider-specific settings (API key and base URL) from the model name."""

    if isinstance(api_key, Path):
        api_key = _get_api_key_from_file(api_key)

    if model_name.startswith("gpt") or model_name.startswith("text-embedding"):
        base_url = LLMProviders.OPENAI
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    elif model_name.startswith("deepseek"):
        base_url = LLMProviders.DEEPSEEK
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    elif model_name.startswith("qwen"):
        base_url = LLMProviders.QWEN
        api_key = api_key or os.getenv("QWEN_API_KEY", "")
    elif model_name.startswith("gemini"):
        base_url = LLMProviders.GEMINI
        api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    else:
        raise ValueError(f"Unsupported model: {model_name}. Litellm will be used as the underlying backend in the next version.")

    return api_key, base_url


def _create_client(api_key: str, **kwargs):
    assert api_key != "", "API key is required."
    client = AsyncOpenAI(api_key=api_key, **kwargs)
    return client


class LLMArguments(BaseModel):
    max_tokens: int = Field(default=512, ge=1, le=16384, description="The maximum number of tokens to generate.")
    temperature: float = Field(default=0.1, ge=0.0, le=1.0, description="The sampling temperature.")
    max_timeouts: int = Field(default=3, ge=1, le=10, description="The maximum number of timeouts.")


class LLMClient:
    default_model: str | None = None
    client = None
    config: LLMArguments = LLMArguments()

    @classmethod
    def configure(
        cls,
        model_name: str,
        api_key: str | Path | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        """
        Configure the shared LLM client.

        The provider (OpenAI / DeepSeek / Qwen) is inferred from ``model_name``,
        and appropriate defaults for ``base_url`` and ``api_key`` are applied.
        Users can still override both ``api_key`` and ``base_url`` explicitly.
        """
        cls.default_model = model_name
        api_key, base_url = _get_openai_compatible_provider_info(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
        )
        cls.client = _create_client(api_key=api_key, base_url=base_url, **kwargs)
        return cls()

    async def create_embedding(self, text: list[str] | str, embed_model: str = "text-embedding-3-large"):
        api_key, base_url = _get_openai_compatible_provider_info(model_name=embed_model)
        self.client = _create_client(api_key=api_key, base_url=base_url)
        response = await self.client.embeddings.create(
            input=text, model=embed_model
        )
        cost = (response.usage.total_tokens / 1000) * MODEL_PRICING[embed_model]["Input"]
        return np.array([data.embedding for data in response.data]).squeeze(), cost

    async def __call__(self,
        messages: list[dict[str, str]],
        parse_tags: bool = False,
        parse_code: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        model_name = kwargs.pop("model", None)
        if model_name is not None:
            api_key, base_url = _get_openai_compatible_provider_info(model_name=model_name)
            self.client = _create_client(api_key=api_key, base_url=base_url)
        else:
            model_name = self.default_model
        timeout = 0
        success = False
        while not success and timeout < self.config.max_timeouts:
            timeout += 1
            try:
                response = await self.client.responses.create(
                    model=model_name,
                    input=messages,
                    max_output_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                llm_output = response.output_text
                token_cost = self._compute_usage(response)
                success = True
            except Exception as e:
                logger.error(f"An error occurs when creating a response: {e}")

        outputs = dict()
        outputs["raw_output"] = llm_output
        if parse_tags:
            tags: list[str] = kwargs["tags"]
            for tag in tags:
                outputs[tag] = self._extract_xml(llm_output, tag)
        elif parse_code:
            code = self._extract_code(llm_output, lang=kwargs["lang"])
            outputs["output"] = code
        else:
            outputs["output"] = llm_output
        outputs["cost"] = token_cost
        return outputs
    
    def _compute_usage(self, response):
        """
        Compute the token cost of an LLM completion.
        
        Args:
            response: The response object from the LLM API.
        
        Returns:
            float: The total token cost.
        """
        model_name = response.model
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cached_tokens = response.usage.input_tokens_details.cached_tokens
        
        # Get the pricing for the model
        pricing = MODEL_PRICING[model_name]
        
        # Compute the cost for each type of token
        # If the model is qwen-series, no price differences are in input and cached tokens
        if model_name.startswith("qwen"):
            input_cost = (input_tokens / 1000) * pricing["Input"]
            output_cost = (output_tokens / 1000) * pricing["Output"]
            return input_cost + output_cost
        else:
            # Otherwise, consider the difference between input and cached tokens
            input_cost = (input_tokens - cached_tokens) / 1000 * pricing["Input"]
            output_cost = (output_tokens / 1000) * pricing["Output"]
            cache_cost = (cached_tokens / 1000) * pricing["Cache"]
            return input_cost + output_cost + cache_cost

    def _extract_xml(self, text: str, tag: str):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_code(self, text: str, lang: str = "python"):
        pattern = rf"```{lang}(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
