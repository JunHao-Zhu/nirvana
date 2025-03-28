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
    client = OpenAI(api_key=api_key, base_url=kwargs["base_url"])
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
    def configure(cls, model_name: str = None, api_key: Union[str, Path] = None, base_url=None):
        cls.model = model_name
        api_key = api_key if isinstance(api_key, str) else _get_api_key_from_file(api_key)
        cls.client = _create_client(api_key, base_url=base_url)
        return cls()
    
    def __call__(
            self,
            messages: List[Dict[str, str]],
            parse_tags: Union[List[str], str] = None,
            **kwargs,
    ) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            **kwargs
        )
        llm_ouput = response.choices[0].message.content

        outputs = dict()
        if isinstance(parse_tags, str):
            outputs[parse_tags] = self._extract_xml(llm_ouput, parse_tags)
        else:
            for tag in parse_tags:
                outputs[tag] = self._extract_xml(llm_ouput, tag)
        return outputs

    def _extract_xml(self, text: str, tag: str):
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else None
