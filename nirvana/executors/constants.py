from enum import Enum


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
    # Deepseek-series models
    "deepseek-chat": {"Input": 0.00028, "Cache": 0.000028, "Output": 0.00042},
    # Qwen-series models
    "qwen-max-latest": {"Input": 0.00033, "Output": 0.0013},
    # Gemini-series models
    "gemini-3-pro": {"Input": 0.002, "Cache": 0.0002, "Output": 0.012},
    "gemini-2.5-pro": {"Input": 0.00125, "Cache": 0.000125, "Output": 0.01},
    "gemini-2.5-flash": {"Input": 0.0003, "Cache": 0.00003, "Output": 0.0025},
}


class LLMProviders(str, Enum):
    DEEPSEEK = "https://api.deepseek.com"
    GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/"
    OPENAI = "https://api.openai.com/v1"
    QWEN = "https://dashscope.aliyuncs.com/compatible-mode/v1"
