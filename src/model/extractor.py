from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer
)


class EmailExtractorModel():
    
    def __init__(
            self,
            model_args
        ):
        self.model_name_or_path = model_args.model_name_or_path
        self.load_finetuned_model = model_args.load_finetuned_model

    def load_tokenizer(
            self
        ):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name_or_path)
        if not self.load_finetuned_model:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        return self.tokenizer

    def load_model(
            self
        ):
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name_or_path)
        self.model.to("cuda")
        if not self.load_finetuned_model:
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        return self.model