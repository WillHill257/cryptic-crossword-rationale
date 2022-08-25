import os

from cryptics.data.format_predictions import format_predictions_as_json

txt_name = "generated_predictions.txt"
json_name = "generated_predictions.json"


def convert(dirpath):
    # open the txt file
    txt = open(f"{dirpath}/{txt_name}", "r")
    lines = txt.readlines()
    txt.close()

    # loop through the text file, extracting the input, labels and predictions
    inputs = []
    labels = []
    predictions = []
    locs_of_sep = []

    for i, line in enumerate(lines):
        # discard the first 3 lines (headers), and the last line (trailing horizontal line)
        if i == 0:
            # use the first line to find the "widths" of each column
            # there are 4 separators -> first line has "+"
            locs_of_sep = [i for i in range(len(line)) if line[i] == "+"]

        if i < 3 or i == len(lines) - 1:
            continue

        # extract the three columns
        inputs.append(line[locs_of_sep[0] + 1 : locs_of_sep[1]].strip())
        labels.append(line[locs_of_sep[1] + 1 : locs_of_sep[2]].strip())
        predictions.append(line[locs_of_sep[2] + 1 : locs_of_sep[3]].strip())

    # write all the data to a json file
    with open(f"{dirpath}/{json_name}", "w", encoding="utf-8") as writer:
        writer.write(
            format_predictions_as_json(
                inputs,
                labels,
                predictions,
            )
        )

        writer.close()


def dir_walk():
    # loop through all of the experiment folders (excluding checkpoints)
    for (dirpath, dirnames, filenames) in os.walk("./experiments", topdown=True):
        # don't process checkpoints
        del_idx = [i for i, x in enumerate(dirnames) if "checkpoint" in x]
        for idx in del_idx[::-1]:
            del dirnames[idx]

        # check that a txt table is present
        # check if a json file is not present -> no change done
        if (txt_name in filenames) and (json_name not in filenames):
            # do the conversion
            print(f"Converting {dirpath}/{txt_name}...")
            convert(dirpath)

        # don't recurse into files
        del filenames[:]

    return


if __name__ == "__main__":
    dir_walk()
