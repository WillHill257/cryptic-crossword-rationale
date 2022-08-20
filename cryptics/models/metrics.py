from types import FunctionType
from typing import Dict, Tuple
import numpy as np

from transformers import T5Tokenizer, EvalPrediction

from arguments import DataTrainingArguments
from models.feature_conversion import decode_for_eval

from datasets import load_metric

metric = load_metric("sacrebleu")


def compute_accuracy(
    eval_preds: EvalPrediction,
    tokenizer: T5Tokenizer,
    data_args: DataTrainingArguments,
) -> Dict[str, float]:
    # decode the predictions and labels
    decoded_preds, decoded_labels = decode_for_eval(eval_preds, tokenizer, data_args)

    # count the number of correct predictions
    count = 0
    for i in range(len(decoded_labels)):
        count += decoded_preds[i] == decoded_labels[i]

    accuracy = count / len(decoded_labels)
    result = {"accuracy": accuracy}
    return result
