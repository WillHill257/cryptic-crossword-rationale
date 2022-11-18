import os
import sys
import json

original_name = "generated_predictions.json"
sorted_name = "sorted_predictions.json"


def sort_predictions(dirpath):
    # open the json file and read all the content into a list of json objects
    original_file = open(f"{dirpath}/{original_name}", "r")
    original_predictions = json.loads(original_file.read())
    original_file.close()

    # original_predictions is an array of Clue dict-like objs
    sorted_predictions = sorted(original_predictions, key=lambda obj: obj["clue"])

    # write to json
    sorted_file = open(f"{dirpath}/{sorted_name}", "w")
    sorted_file.write(json.dumps(sorted_predictions))
    sorted_file.close()


def main():
    # loop through all of the experiment folders (excluding checkpoints)
    for (dirpath, dirnames, filenames) in os.walk("./experiments/", topdown=True):
        # don't process checkpoints
        del_idx = [i for i, x in enumerate(dirnames) if "checkpoint" in x]
        for idx in del_idx[::-1]:
            del dirnames[idx]

        # check if a generate_predictions.json file is present
        if (original_name in filenames) and (sorted_name not in filenames):
            # sort the predictions
            print(f"Sorting {dirpath}/{original_name}...")
            sort_predictions(dirpath)

        # don't recurse into files
        del filenames[:]

    return


if __name__ == "__main__":
    main()
