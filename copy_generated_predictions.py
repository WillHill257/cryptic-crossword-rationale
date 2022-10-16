"""
copy the generated predictions (from the experimental output to the data folder)
"""

import os
import json

if __name__ == "__main__":
    # define the path components
    experimental_path = "./experiments/i_or_data"
    data_path = "./cryptics/data/json/generated-predictions"
    dataset_names = ["random", "naive-disjoint", "word-initial-disjoint"]
    models = ["t5-small", "t5-large"]
    splits = ["train", "validation", "test"]
    rationale_filename = "generated_predictions.json"

    # loop through all the files
    for dataset in dataset_names:
        for model in models:
            for split in splits:
                # determine the source and destinations, with the filenames
                source_path = f"{experimental_path}/{dataset}/{model}/{split}/{rationale_filename}"
                destination_path = f"{data_path}/{dataset}/{model}/{split}.json"

                print(f"Copying {source_path} to {destination_path}")

                # make sure the destination directory exists
                i = len(destination_path) - destination_path[::-1].find("/")
                os.makedirs(os.path.dirname(destination_path[:i]), exist_ok=True)

                # read the file and convert it to json
                with open(source_path, "r") as source_file:
                    json_obj = json.loads(source_file.read())

                # write the destination file, with the added "data" field
                json_obj = {"data": json_obj}
                with open(destination_path, "w") as dest_file:
                    dest_file.write(json.dumps(json_obj))
