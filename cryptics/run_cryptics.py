# heavily based on https://github.com/huggingface/transformers/blob/v4.21.1/examples/pytorch/translation/run_translation.py

import logging
import os
import sys

import datasets

import transformers
from transformers import (
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

from arguments import DataTrainingArguments, ModelArguments
from models.metrics import compute_accuracy
from data.load_data import load_data
from models.feature_conversion import (
    decode_features,
    encode_features,
    feature_conversion,
    feature_conversion_curriculum,
)
from data.format_predictions import format_predictions_as_json

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.0")

logger = logging.getLogger(__name__)


def determine_experiment(
    experiment_folder: str, training_args: Seq2SeqTrainingArguments
):
    # calculate the new dir
    new_dir = training_args.output_dir + "/" + experiment_folder

    # make sure slash isn't doubled
    new_dir = new_dir.replace("//", "/")

    # create the new dir
    os.makedirs(new_dir, exist_ok=True)

    return new_dir


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 3 and sys.argv[2].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[2])
        )
    else:
        print(f"Usage: python3 {sys.argv[0]} experiment_folder configuration_file")
        print(f"Example: python3 {sys.argv[0]} 0001/a configurations/config.json")
        exit()

    # save the experiment info in a new folder
    training_args.output_dir = determine_experiment(sys.argv[1], training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 3:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behaviour, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load datasets
    raw_datasets = load_data(data_args.dataset_name, model_args.cache_dir)
    logger.info(f"Loaded the {data_args.dataset_name} dataset:\n{raw_datasets}")

    # two lambda functions for if curriculum learning is done or not
    def conversion_without_curriculum(x, is_inference):
        return feature_conversion(
            x,
            data_args.predict_rationale,
            data_args.include_predicted_rationale_as_input,
            is_inference,
        )

    def conversion_with_curriculum(x, do_descramble: bool):
        return feature_conversion_curriculum(x, do_descramble)

    # format the data as required
    old_columns = raw_datasets["train"].column_names
    with training_args.main_process_first(desc="feature conversion"):
        if not data_args.perform_curriculum_learning:
            for split in ["train", "validation", "test"]:
                if data_args.infer_on_split == split:
                    is_inference = True
                else:
                    is_inference = False

                raw_datasets[split] = raw_datasets[split].map(
                    lambda x: conversion_without_curriculum(x, is_inference),
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=old_columns,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Performing feature conversion",
                )
        else:
            raw_datasets = raw_datasets.map(
                lambda x: conversion_with_curriculum(
                    x, data_args.curriculum_learning_do_descramble
                ),
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=old_columns,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Performing curriculum feature conversion - {}".format(
                    "descramble"
                    if data_args.curriculum_learning_do_descramble
                    else "definition lookup"
                ),
            )

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = T5Config.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    # ensure decoder_start_token_id is defined
    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    # lambda function for ease of use
    preprocess_function = lambda data: encode_features(
        data["input"], data["label"], tokenizer, data_args
    )

    # tokenize the training dataset
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    # tokenize the validation dataset
    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    # tokenize the testing dataset
    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")

        predict_dataset = raw_datasets[data_args.infer_on_split]

        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    compute_metrics = lambda data: compute_accuracy(data, tokenizer, data_args)

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if training_args.predict_with_generate
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload and for loading later as a pretrained model

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=max_length,
            num_beams=num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = decode_features(predict_results.predictions, tokenizer)
                prediction_input = dict()
                prediction_input["input"] = decode_features(
                    predict_dataset["input_ids"], tokenizer
                )
                prediction_input["label"] = decode_features(
                    predict_dataset["labels"], tokenizer
                )
                predictions = [pred.strip() for i, pred in enumerate(predictions)]
                output_prediction_file = os.path.join(
                    training_args.output_dir, "generated_predictions.json"
                )
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write(
                        format_predictions_as_json(
                            prediction_input["input"],
                            prediction_input["label"],
                            predictions,
                        )
                    )

    # flush any buffers
    logging.shutdown()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
