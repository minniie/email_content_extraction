from argparse import ArgumentTypeError

from transformers import Trainer

from src.dataset.enron import EnronDataset, EnronCollator
from src.dataset.squad import SquadDataset, SquadCollator
from src.dataset.princeton_email import PrincetonEmailDataset, PrincetonEmailCollator
from src.model.extractor import EmailExtractorModel
from src.util.args import set_args
from src.util.metric import MetricCallback


def main():
    # set arguments
    model_args, data_args, training_args = set_args()

    # load model
    model_cls = EmailExtractorModel(model_args)
    tokenizer = model_cls.load_tokenizer()
    model = model_cls.load_model()

    # load dataset and collator
    if data_args.dataset_name == "enron":
        dataset_cls, collator_cls = EnronDataset, EnronCollator
    elif data_args.dataset_name == "squad":
        dataset_cls, collator_cls = SquadDataset, SquadCollator
    elif data_args.dataset_name == "princeton_email":
        dataset_cls, collator_cls = PrincetonEmailDataset, PrincetonEmailCollator
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
        data_collator=collator,
        callbacks=[MetricCallback]
    )
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
    if training_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()