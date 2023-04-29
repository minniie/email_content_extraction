from dataclasses import dataclass, field

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)


@dataclass
class ModelArguments:
    """
    Arguments for models
    """
    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "model name or path"
        }
    )
    load_finetuned_model: bool = field(
        default=False,
        metadata={
            "help": "whether to load finetuned model"
        }
    )


@dataclass
class DataArguments:
    """
    Arguments for datasets
    """
    dataset_name: str = field(
        default=None,
        metadata={
            "help": "dataset name"
        }
    )


def set_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args,training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    return model_args, data_args, training_args