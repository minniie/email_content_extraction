from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)


class EmailExtractorModel():
    def __init__(
            self,
            model_name_or_path
        ):
        self.model_name_or_path = model_name_or_path

    def load_tokenizer(
            self
        ):
        tokenizer = GPT2Tokenizer.from_pretrained(self.model_name_or_path)
        return tokenizer

    def load_model(
            self
        ):
        model = GPT2LMHeadModel.from_pretrained(self.model_name_or_path)
        return model
        