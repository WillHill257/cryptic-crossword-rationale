from typing import Dict, List, Tuple
from data.clue import Clue
from arguments import DataTrainingArguments
import numpy as np
import random

from transformers import T5Tokenizer

# random number generator
rng = random.Random()
rng.seed(42)

# function to scramble the letters of the provided word
def randomise_letters(word: str) -> str:
    x = list(word)
    rng.shuffle(x)
    return "".join(x)


def feature_conversion(
    clue_dict: Dict,
    predict_rationale: bool,
    include_rationale: bool,
    is_inference: bool,
) -> Dict[str, str]:
    # create a Clue object
    clue_obj = Clue(
        clue=clue_dict["clue"],
        answer=clue_dict["answer"],
        annotation=clue_dict["annotation"],
        predicted_rationale=clue_dict["predicted_rationale"],
        predicted_answer=clue_dict["predicted_answer"],
    )

    # convert the item to the correct format
    return clue_obj.convert_to_feature(
        predict_rationale, include_rationale, is_inference
    )


def feature_conversion_curriculum(
    data_dict: Dict, descramble_task: bool
) -> Dict[str, str]:
    """convert the object to an input and label string pairing"""

    # Example input dict: {"idx": 0, "input": "Litigator's group (3)", "target": "aba"}
    # output if no scrambling: {input: "phrase: Litigator's group (3)" , label: "aba"}
    # output if do scrambling: {input: "descramble: baa Litigator's group (3)" , label: "aba"} -> scrambled version can be randomly at the beginning or the end

    # build the input strings
    if not descramble_task:
        # is the definition lookup task
        # e.g. {input: "phrase: Litigator's group (3)" , label: "aba"}
        input_string = f"phrase: {data_dict['input']}"
    else:
        # is the descrambling task
        # e.g. {input: "descramble: baa Litigator's group (3)" , label: "aba"}

        # start with the clue
        input_string = data_dict["input"]

        # randomly add the scrambled version to the beginning or end
        random_word = randomise_letters(data_dict["target"])
        if rng.random() > 0.5:
            # add to the front
            input_string = random_word + " " + input_string
        else:
            # add to the end
            input_string = input_string + " " + random_word

        # add the prompt
        input_string = "descramble: " + input_string

    label_string = f"{data_dict['target']}"

    return {"input": input_string, "label": label_string}


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


def extract_answer(label: str) -> str:
    # find the location of the word "explanation" - this is the marker for the rationale
    idx = label.lower().find("explanation:")

    if idx < 0:
        # no explanation given
        return label
    else:
        # extract everything before the rationale
        return label[:idx]


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
    decoded_preds = [extract_answer(pred).strip().upper() for pred in decoded_preds]
    decoded_labels = [extract_answer(label).strip().upper() for label in decoded_labels]

    return decoded_preds, decoded_labels


def decode_features(features: List[str], tokenizer: T5Tokenizer):
    return tokenizer.batch_decode(
        features,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
