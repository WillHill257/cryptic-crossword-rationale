from typing import List
import json

from data.clue import Clue


def format_predictions_as_json(
    inputs: List[str], labels: List[str], predictions: List[str]
) -> str:
    """
    have each data item as a json object, all bundled together into an array
    """

    # loop through each data item
    items = []
    for input, label, prediction in zip(inputs, labels, predictions):
        # create an empty clue
        clue = Clue("", "", "")

        # reconstruct the clue format
        clue.convert_from_feature(input, label, prediction)

        # add the map to the list
        items.append(clue.to_map())

    # return a json-encoded string
    return json.dumps(items)


def format_predictions_as_table(
    inputs: List[str], labels: List[str], predictions: List[str]
) -> str:
    """
    have a table:
        +---------+--------+------------+
        | input   | label  | prediction |
        +---------+--------+------------+
    """

    # find the longest input, label, prediction
    max_input = 0
    max_label = 0
    max_prediction = 0
    for i in range(len(inputs)):
        max_input = max(max_input, len(inputs[i]))
        max_label = max(max_label, len(labels[i]))
        max_prediction = max(max_prediction, len(predictions[i]))

    # add some padding to the end of each
    max_input += 10
    max_label += 10
    max_prediction += 10

    # add the top line, and headings
    input_separator = "-" * max_input
    label_separator = "-" * max_label
    prediction_separator = "-" * max_prediction

    pad_input = lambda x: x + " " * (max_input - len(x))
    pad_label = lambda x: x + " " * (max_label - len(x))
    pad_prediction = lambda x: x + " " * (max_prediction - len(x))

    horizontal_separator = (
        f"+{input_separator}+{label_separator}+{prediction_separator}+\n"
    )

    output = horizontal_separator
    output += "|{}|{}|{}|\n".format(
        pad_input(" input"), pad_label(" label"), pad_prediction(" prediction")
    )
    output += horizontal_separator

    # add the data
    for i in range(len(inputs)):
        output += "|{}|{}|{}|\n".format(
            pad_input(" " + inputs[i]),
            pad_label(" " + labels[i]),
            pad_prediction(" " + predictions[i]),
        )

    # add the bottom line
    output += horizontal_separator

    return output
