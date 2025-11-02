import logging
import re
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


MODEL_PRICING = { # pricing policies (in US dollar per 1k tokens)
    # models from OpenAI
    "gpt-5-2025-08-07": {"Input": 0.00125, "Cache": 0.000125, "Output": 0.01},
    "gpt-5-mini-2025-08-07": {"Input": 0.00025, "Cache": 0.000025, "Output": 0.002},
    "gpt-5-nano-2025-08-07": {"Input": 0.00005, "Cache": 0.000005, "Output": 0.0004},
    "gpt-4.1-2025-04-14": {"Input": 0.002, "Cache": 0.0005, "Output": 0.008},
    "gpt-4.1-mini-2025-04-14": {"Input": 0.0004, "Cache": 0.0001, "Output": 0.0016},
    "gpt-4.1-nano-2025-04-14": {"Input": 0.0001, "Cache": 0.000025, "Output": 0.0004},
    "gpt-4o-2024-08-06": {"Input": 0.0025, "Cache": 0.00125, "Output": 0.01},
    "gpt-4o-mini-2024-07-18": {"Input": 0.00015, "Cache": 0.000075, "Output": 0.0006},
    "text-embedding-3-large": {"Input": 0.00013,},
    # models from DeepSeek
    "deepseek-chat": {"Input": 0.00027, "Cache": 0.00007, "Output": 0.0011},
    # models from Qwen
    "qwen-max-latest": {"Input": 0.00033, "Output": 0.0013},
}


def _get_api_key_from_file(file):
    with open(file, 'r') as api_file:
        api_key = api_file.readline()
    return api_key


def _create_client(api_key, **kwargs):
    assert api_key != "", "API key is required."
    client = AsyncOpenAI(api_key=api_key, **kwargs)
    return client


@dataclass
class LLMArguments:
    max_tokens: int = field(default=512, metadata={"help": "The maximum number of tokens to generate."})
    temperature: float = field(default=0.1, metadata={"help": "The sampling temperature."})
    max_timeouts: int = field(default=3, metadata={"help": "The maximum number of timeouts."})


class LLMClient:
    default_model: Optional[str] = None
    client = None
    config: LLMArguments = LLMArguments()

    @classmethod
    def configure(cls, model_name: str = None, api_key: Union[str, Path] = None, **kwargs):
        cls.default_model = model_name
        api_key = api_key if isinstance(api_key, str) else _get_api_key_from_file(api_key)
        cls.client = _create_client(api_key, **kwargs)
        return cls()

    async def create_embedding(self, text: Union[list[str], str], embed_model: str = "text-embedding-3-large"):
        response = await self.client.embeddings.create(
            input=text, model=embed_model
        )
        cost = (response.usage.total_tokens / 1000) * MODEL_PRICING[embed_model]["Input"]
        return np.array([data.embedding for data in response.data]).squeeze(), cost

    async def __call__(self,
            messages: List[Dict[str, str]],
            parse_tags: bool = False,
            parse_code: bool = False,
            **kwargs,
    ) -> Dict[str, Any]:
        model_name = kwargs.get("model", self.default_model)
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
        if parse_tags:
            tags: List[str] = kwargs["tags"]
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
