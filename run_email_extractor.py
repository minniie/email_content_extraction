from argparse import ArgumentTypeError

from transformers import Trainer

from src.dataset.emailsum import EmailSumDataset
from src.dataset.squad import SquadDataset
from src.model.extractor import EmailExtractorModel
from src.util.args import set_args


def main():
    # set arguments
    model_args, data_args, training_args = set_args()

    # load dataset
    if data_args.dataset_name == "emailsum":
        dataset_cls_name = EmailSumDataset
    elif data_args.dataset_name == "squad":
        dataset_cls_name = SquadDataset
    else:
        raise ArgumentTypeError(f"dataset not supported: {data_args.dataset_name}")
    dataset_cls = dataset_cls_name()
    dataset = dataset_cls.load()

    # load model
    model_cls = EmailExtractorModel(model_args.model_name_or_path)
    tokenizer = model_cls.load_tokenizer()
    model = model_cls.load_model()

    # load trainer
    trainer =  Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()