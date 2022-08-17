from types import FunctionType
from typing import Tuple
import numpy as np

from transformers import T5Tokenizer

from arguments import DataTrainingArguments
from models.feature_conversion import decode_for_eval

from datasets import load_metric

metric = load_metric("sacrebleu")


def compute_accuracy(
    eval_preds: Tuple,
    tokenizer: T5Tokenizer,
    postprocessor: FunctionType,
    data_args: DataTrainingArguments,
):
    # decode the predictions and labels
    decoded_preds, decoded_labels = decode_for_eval(eval_preds, tokenizer, data_args)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    # prediction_lens = [
    #     np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    # ]
    # result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
