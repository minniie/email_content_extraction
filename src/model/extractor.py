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
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        return self.tokenizer

    def load_model(
            self
        ):
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name_or_path)
        self.model.to("cuda")
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        return self.model