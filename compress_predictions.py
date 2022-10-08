import zipfile
import os
import sys

original_name = "generated_predictions.json"
zipped_name = "generated_predictions.json.zip"


def compress(dirpath, filenames):
    # check if a generate_predictions.json file is present
    if (original_name in filenames) and (zipped_name not in filenames):
        # need to compress this file
        print(f"Compressing {dirpath}/{original_name}...")
        zipped = zipfile.ZipFile(f"{dirpath}/{zipped_name}", "w")
        zipped.write(
            f"{dirpath}/{original_name}",
            original_name,
            compress_type=zipfile.ZIP_DEFLATED,
        )
        zipped.close()


def uncompress(dirpath, filenames):
    # check if a generate_predictions.json file is present
    if zipped_name in filenames:
        # need to compress this file
        print(f"Uncompressing {dirpath}/{zipped_name}...")
        zipped = zipfile.ZipFile(f"{dirpath}/{zipped_name}", "r")
        zipped.extractall(f"{dirpath}")
        zipped.close()


def main(compression_type):
    # loop through all of the experiment folders (excluding checkpoints)
    for (dirpath, dirnames, filenames) in os.walk(
        "./experiments/i_or_data/", topdown=True
    ):
        # don't process checkpoints
        del_idx = [i for i, x in enumerate(dirnames) if "checkpoint" in x]
        for idx in del_idx[::-1]:
            del dirnames[idx]

        if compression_type == "compress":
            compress(dirpath, filenames)
        else:
            uncompress(dirpath, filenames)

        # don't recurse into files
        del filenames[:]

    return


if __name__ == "__main__":
    # check the sysargs
    options = ["compress", "uncompress"]
    if len(sys.argv) != 2 or sys.argv[1] not in options:
        print(f"Usage: {sys.argv[0]} <compression type: {options[0]} or {options[1]}>")
        exit()

    main(sys.argv[1])
