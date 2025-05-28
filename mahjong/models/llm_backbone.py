import logging
import re
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


MODEL_PRICING = { # pricing policies (in US dollar per 1k tokens)
    # models from OpenAI
    "gpt-4.1-2025-04-14": {"Input": 0.002, "Output": 0.008},
    "gpt-4.1-mini-2025-04-14": {"Input": 0.0004, "Output": 0.0016},
    "gpt-4.1-nano-2025-04-14": {"Input": 0.0001, "Output": 0.0004},
    "gpt-4o-2024-08-06": {"Input": 0.0025, "Output": 0.01},
    "gpt-4o-mini-2024-07-18": {"Input": 0.00015, "Output": 0.0006},
    "text-embedding-3-large": {"Input": 0.00013,},
    # models from DeepSeek
    "deepseek-chat": {"Input": 0.00007, "Output": 0.0011},
}


def _get_api_key_from_file(file):
    with open(file, 'r') as api_file:
        api_key = api_file.readline()
    return api_key


def _create_client(api_key, **kwargs):
    assert api_key != "", "API key is required."
    client = OpenAI(api_key=api_key, **kwargs)
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

    def create_embedding(self, text: Union[list[str], str], embed_model: str = "text-embedding-3-large"):
        response = self.client.embeddings.create(
            input=text, model=embed_model
        )
        cost = (response.usage.total_tokens / 1000) * MODEL_PRICING[embed_model]["Input"]
        return np.array([data.embedding for data in response.data]).squeeze(), cost
    
    def __call__(
            self,
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
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                llm_output = response.choices[0].message.content
                input_cost = (response.usage.prompt_tokens / 1000) * MODEL_PRICING[model_name]["Input"]
                output_cost = (response.usage.completion_tokens / 1000) * MODEL_PRICING[model_name]["Output"]
                token_cost = input_cost + output_cost
                success = True
            except Exception as e:
                logger.error(f"Timeout errors.")

        outputs = dict()
        if parse_tags:
            tags: List[str] = kwargs["tags"]
            for tag in tags:
                outputs[tag] = self._extract_xml(llm_output, tag)
        elif parse_code:
            code = self._extract_code(llm_output, lang=kwargs["lang"])
            outputs["output"] = code
        else:
            raise ValueError("Specify parsing type for llm outputs: either parsing tags or code.")
        outputs["cost"] = token_cost
        return outputs

    def _extract_xml(self, text: str, tag: str):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_code(self, text: str, lang: str = "python"):
        pattern = rf"```{lang}(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
