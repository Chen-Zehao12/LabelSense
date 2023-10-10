from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    alpha: Optional[int] = field(
        default=1, metadata={"help": "The weight of Similarity loss."}
    )

    beta: Optional[int] = field(
        default=1, metadata={"help": "The weight of Classification loss."}
    )

    gamma: Optional[int] = field(
        default=1, metadata={"help": "The weight of Cosine Embedding loss."}
    )

    momentum_rate: Optional[float] = field(
        default=0.0, metadata={"help": "The rate of Momentum update."}
    )

    centroid: Optional[bool] = field(
        default=False, metadata={"help": "If use centroid"}
    )

    cls_loss: Optional[str] = field(
        default='CrossEntropyLoss', metadata={"help": "If use BCELoss else CrossEntropyLoss"}
    )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    subsampling_rate: float = field(
        default=1.0, metadata={"help": "The rate of subsampling, default is 1.0 (Not subsampling)."}
    )

    stratified_sampling: bool = field(
        default=False, metadata={"help": "If stratified sampling else random sampling."}
    )

    with_example: bool = field(
        default=True, metadata={"help": "Semantic label with one sentence example."}
    )

    text_max_length: Optional[int] = field(
        default=512, metadata={"help": "The maximum total input sequence length after tokenization."}
    )

    num_proc: Optional[int] = field(
        default=8, metadata={"help": "The num proc in dataset.map()."}
    )