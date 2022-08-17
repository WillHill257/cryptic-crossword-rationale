# create a clue class

from typing import *


class Clue:
    def __init__(
        self, clue: str, answer: str, annotation: str, predicted_rationale: str = ""
    ) -> None:
        self.clue = clue
        self.answer = answer
        self.annotation = annotation
        self.predicted_rationale = predicted_rationale

    def to_map(self) -> str:
        """convert the Clue to an object representation, so it can be encoded as JSON"""
        return {
            "clue": self.clue,
            "answer": self.answer,
            "annotation": self.annotation,
            "rationale": self.predicted_rationale,
        }

    def convert_to_feature(
        self, predict_rationale: bool, include_rationale: bool
    ) -> str:
        """generate the strings to use as input and label for T5"""

        # the clue is always part of the input string
        in_string = "clue: " + self.clue

        # the label is always part of the label string
        label_string = self.answer

        # if we need to predict the rationale as well, prepend the prompt to the input and the rationale to the label
        # only do this if there is an annotation to predict
        if predict_rationale and self.annotation != "":
            # prepend the prompt
            in_string = "explain " + in_string

            # add the rationale to the label
            label_string = label_string + " explanation: " + self.annotation

        # if we want to use the previously-predicted rationale as input
        if include_rationale:
            # no extra prompt is prepended
            # append the predicted rationale
            in_string = in_string + " explanation: " + self.predicted_rationale

            # no change is made to the label string

        return {"input": in_string, "label": label_string}


# clue = Clue(
#     "the middle of this sentence",
#     "of",
#     "'of' is the middle word",
#     "the middle word is 'of'",
# )
# print(clue.convert_to_feature(True, False))  # I -> OR
# print(clue.convert_to_feature(False, True))  # IR -> O
# print(clue.convert_to_feature(False, False))  # I -> O
# print(clue.convert_to_feature(True, True))  # IR -> OR' (shouldn't be used)
