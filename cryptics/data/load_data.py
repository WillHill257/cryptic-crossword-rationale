from typing import Union
from datasets.dataset_dict import DatasetDict, IterableDatasetDict
from datasets.iterable_dataset import IterableDataset
from datasets.arrow_dataset import Dataset

from datasets import load_dataset

datasets_names = ["random", "naive-disjoint", "word-initial-disjoint"]


def load_data(
    dataset: str, cache_dir: str
) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
    """function to load the train and validation splits of the specified dataset, which should be one of "random", "naive-disjoint", or "word-initial-disjoint" """

    # if the dataset isn't valid, raise an error
    if dataset not in datasets_names:
        raise ValueError(
            "The specified value for 'dataset' is not supported. "
            "Please choose one of {}".format(datasets_names)
        )

    # set the base path
    base_path = f"data/json/{dataset}/"

    # load the train, validation, and test splits
    dataset = load_dataset(
        "json",
        data_files={
            "train": base_path + "train.json",
            "validation": base_path + "validation.json",
            "test": base_path + "test.json",
        },
        field="data",
        cache_dir=cache_dir,
    )

    return dataset
