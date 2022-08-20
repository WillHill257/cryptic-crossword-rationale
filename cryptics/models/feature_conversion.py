from typing import List, Tuple
from data.clue import Clue
from arguments import DataTrainingArguments
import numpy as np

from transformers import T5Tokenizer


def feature_conversion(
    clue_dict: dict, predict_rationale: bool, include_rationale: bool
) -> str:
    # create a Clue object
    clue_obj = Clue(
        clue=clue_dict["clue"],
        answer=clue_dict["answer"],
        annotation=clue_dict["annotation"],
        predicted_rationale=clue_dict["rationale"],
    )

    # convert the item to the correct format
    return clue_obj.convert_to_feature(predict_rationale, include_rationale)


def encode_features(
    inputs: List[str],
    targets: List[str],
    tokenizer: T5Tokenizer,
    data_args: DataTrainingArguments,
):
    """tokenize the input and label strings"""

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    # tokenise the input and label strings
    model_inputs = tokenizer(
        inputs,
        max_length=data_args.max_source_length,
        padding=padding,
        truncation=True,
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, padding=padding, truncation=True
        )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    # combine them into a single entity
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def decode_for_eval(
    eval_preds: Tuple, tokenizer: T5Tokenizer, data_args: DataTrainingArguments
) -> Tuple[List[str], List[str]]:
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip().upper() for pred in decoded_preds]
    decoded_labels = [label.strip().upper() for label in decoded_labels]

    return decoded_preds, decoded_labels


def decode_features(features: List[str], tokenizer: T5Tokenizer):
    return tokenizer.batch_decode(
        features,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
