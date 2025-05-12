import re
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from dataclasses import dataclass, field
from openai import OpenAI

    
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


class LLMClient:
    model: Optional[str] = None
    client = None
    config: LLMArguments = LLMArguments()

    @classmethod
    def configure(cls, model_name: str = None, api_key: Union[str, Path] = None, **kwargs):
        cls.model = model_name
        api_key = api_key if isinstance(api_key, str) else _get_api_key_from_file(api_key)
        cls.client = _create_client(api_key, **kwargs)
        return cls()
    
    def __call__(
            self,
            messages: List[Dict[str, str]],
            parse_tags: bool = False,
            parse_code: bool = False,
            **kwargs,
    ) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )
        llm_output = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        # according to pricing policies from OpenAI, DeepSeek and QWen, cost_per_output_token = 4 * cost_per_input_token
        token_cost = input_tokens + 4 * output_tokens

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
