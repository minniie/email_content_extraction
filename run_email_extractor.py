from argparse import ArgumentTypeError

from transformers import Trainer

from src.dataset.emailsum import EmailSumDataset, EmailSumCollator
from src.dataset.squad import SquadDataset, SquadCollator
from src.model.extractor import EmailExtractorModel
from src.util.args import set_args


def main():
    # set arguments
    model_args, data_args, training_args = set_args()

    # load model
    model_cls = EmailExtractorModel(model_args.model_name_or_path)
    tokenizer = model_cls.load_tokenizer()
    model = model_cls.load_model()

    # load dataset and collator
    if data_args.dataset_name == "emailsum":
        dataset_cls, collator_cls = EmailSumDataset, EmailSumCollator
    elif data_args.dataset_name == "squad":
        dataset_cls, collator_cls = SquadDataset, SquadCollator
    else:
        raise ArgumentTypeError(f"dataset not supported: {data_args.dataset_name}")
    dataset = dataset_cls().load()
    collator = collator_cls(tokenizer)

    # load trainer
    trainer =  Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()