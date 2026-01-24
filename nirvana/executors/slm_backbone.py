from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalLMClient:
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    
    @classmethod
    def configure(cls, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        cls.tokenizer = tokenizer
        cls.model = model
        return cls
    
    def __call__(self, *args, **kwargs):
        pass
