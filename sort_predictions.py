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


def find_match():
    ran = "./experiments/random/t5-large/cl/i_or/sorted_predictions.json"
    wid = "./experiments/word-initial-disjoint/t5-large/cl/i_or/sorted_predictions.json"

    # open both files
    ran_file = open(ran, "r")
    ran_predictions = json.loads(ran_file.read())
    ran_file.close()
    wid_file = open(wid, "r")
    wid_predictions = json.loads(wid_file.read())
    wid_file.close()

    ran_i = 0
    wid_i = 0

    while ran_i < len(ran_predictions) and wid_i < len(wid_predictions):
        ran_clue = ran_predictions[ran_i]["clue"]
        wid_clue = wid_predictions[wid_i]["clue"]
        # if they are the same, print
        if ran_clue == wid_clue:
            ran_right = (
                ran_predictions[ran_i]["answer"]
                == ran_predictions[ran_i]["predicted_answer"]
            )
            wid_right = (
                wid_predictions[wid_i]["answer"]
                == wid_predictions[wid_i]["predicted_answer"]
            )

            if ran_right:
                ran_predictions[ran_i]["correct"] = True

            if wid_right:
                wid_predictions[wid_i]["correct"] = True

            print(json.dumps(ran_predictions[ran_i]))
            print(json.dumps(wid_predictions[wid_i]))
            print("-" * 50)

            ran_i += 1
            wid_i += 1
        elif ran_clue < wid_clue:
            # ran comes first in alphabet, therefore increase it
            ran_i += 1
        else:
            wid_i += 1


if __name__ == "__main__":
    if sys.argv[1] == "sort":
        main()
    else:
        find_match()
