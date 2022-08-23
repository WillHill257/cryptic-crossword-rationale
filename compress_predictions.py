import zipfile
import os

original_name = "generated_predictions.txt"
zipped_name = "generated_predictions.txt.zip"


def main():
    # loop through all of the experiment folders (excluding checkpoints)
    for (dirpath, dirnames, filenames) in os.walk("./experiments", topdown=True):
        # don't process checkpoints
        del_idx = [i for i, x in enumerate(dirnames) if "checkpoint" in x]
        for idx in del_idx[::-1]:
            del dirnames[idx]

        # check if a generate_predictions.txt file is present
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

        # don't recurse into files
        del filenames[:]

    return


if __name__ == "__main__":
    main()
