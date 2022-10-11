"""
Process the cryptonite-official-splits into a format we require:
    - only include the clue and answer fields
    - reformat the file to be normal json, with all the data under a "data" field (as with the other files)
"""

import json
import os

from create_splits import write_splits_to_json
from clue import Clue

if __name__ == "__main__":

    # the list of input files
    files = {
        "train": "cryptonite-train.jsonl",
        "validation": "cryptonite-val.jsonl",
        "test": "cryptonite-test.jsonl",
    }

    # the data dict
    data = {"train": list(), "validation": list(), "test": list()}

    # create the new folder
    new_location = "./json/cryptonite"
    os.makedirs(new_location, exist_ok=True)

    # for each of these files, filter their content and create the new files
    for split, filename in files.items():
        print(f"Processing {filename}")
        # read each line and parse as dict
        data[split] = [
            json.loads(line)
            for line in open(
                f"./json/cryptonite-official-split/{filename}", "r", encoding="utf-8"
            )
        ]

        # these are the unwanted keys
        # unwanted = set(data[0]) - wanted_keys

        # loop through each line
        for i, line in enumerate(data[split]):
            # create a Clue object, using only the clue and answer
            data[split][i] = Clue(line["clue"], line["answer"], "")

    # write this processed data to a new file
    print("Writing JSON file")
    write_splits_to_json("cryptonite", data["train"], data["validation"], data["test"])
    print("Done")
