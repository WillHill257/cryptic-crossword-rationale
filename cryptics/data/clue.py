# create a clue class

from typing import *


def extract_clue(input: str) -> Tuple[str, str]:
    """extract the clue from a prompt-formatted input string"""

    # find the location of the word "clue:"
    clue_idx = input.lower().find("clue:")

    # find the location of the word "explanation:"
    explanation_idx = input.lower().find("explanation:")

    # account for this prompt not existing
    if explanation_idx < 0:
        explanation_idx = len(input)

    # return everything in between, and after
    return (
        input[clue_idx + len("clue:") : explanation_idx].strip(),
        input[explanation_idx + len("explanation:") :].strip(),
    )


def extract_answer_and_rationale(string: str) -> Tuple[str, str]:
    """extract the answer and rationale from the prompt-formatted string"""

    # find the location of the word "explanation:"
    explanation_idx = string.lower().find("explanation:")

    # account for annotation not existing
    if explanation_idx < 0:
        explanation_idx = len(string)

    # return everything and before and after separately
    return (
        string[:explanation_idx].strip(),
        string[explanation_idx + len("explanation:") :].strip(),
    )


class Clue:
    def __init__(
        self,
        clue: str,
        answer: str,
        annotation: str,
        predicted_rationale: str = "",
        predicted_answer: str = "",
    ) -> None:
        self.clue = clue
        self.answer = answer
        self.annotation = annotation
        self.predicted_rationale = predicted_rationale
        self.predicted_answer = predicted_answer

    def to_map(self) -> str:
        """convert the Clue to an object representation, so it can be encoded as JSON"""
        return {
            "clue": self.clue,
            "answer": self.answer,
            "annotation": self.annotation,
            "predicted_rationale": self.predicted_rationale,
            "predicted_answer": self.predicted_answer,
        }

    def convert_from_feature(self, input: str, label: str, prediction: str) -> None:
        """take the output of the model and recreate a clue object"""

        # extract the answer and annotation (if applicable) from the label
        self.answer, self.annotation = extract_answer_and_rationale(label)

        # extract the predicted answer and rationale (if applicable) form the prediction
        self.predicted_answer, self.predicted_rationale = extract_answer_and_rationale(
            prediction
        )

        # extract the clue from the input string, as well as a potential predicted rationale (if IR->O model is used)
        self.clue, potential_rationale = extract_clue(input)

        if potential_rationale != "":
            self.predicted_rationale = potential_rationale

    def convert_to_feature(
        self,
        predict_rationale: bool,
        include_rationale: bool,
        is_inference: bool,
        use_gold_annotations_with_input: bool,
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
        elif predict_rationale and is_inference:
            # prepend the prompt
            in_string = "explain " + in_string

            # don't need to alter the label, since we do not use it, so self.annotation == "" doesn't matter

        # if we want to use the previously-predicted rationale as input
        if include_rationale:
            # no extra prompt is prepended
            # append the predicted rationale or annotation
            rationale = (
                self.annotation
                if use_gold_annotations_with_input
                else self.predicted_rationale
            )

            if rationale != "":
                in_string = in_string + " explanation: " + rationale

            # no change is made to the label string

        return {"input": in_string, "label": label_string}


# clue = Clue(
#     "the middle of this sentence",
#     "of",
#     "'of' is the middle word",
#     "the middle word is 'of'",
# )
# print(clue.convert_to_feature(True, False, True, False))  # I -> OR
# print(clue.convert_to_feature(False, True, True, True))  # IR -> O
# print(clue.convert_to_feature(False, False, True, False))  # I -> O
# print(clue.convert_to_feature(True, True))  # IR -> OR' (shouldn't be used)

# # I->O
# clue = Clue("", "", "")
# clue.convert_from_feature(
#     "clue: how are you",
#     "GOOD",
#     "BAD",
# )
# print(clue.to_map())
# # I->OR
# clue = Clue("", "", "")
# clue.convert_from_feature(
#     "explain clue: how are you",
#     "GOOD explanation: it is emotion",
#     "BAD explanation: oh well",
# )
# print(clue.to_map())
# # IR->O
# clue = Clue("", "", "")
# clue.convert_from_feature(
#     "clue: how are you explanation: it is an emotion",
#     "GOOD",
#     "BAD",
# )
# print(clue.to_map())
# # IR->OR' (should NOT be used)
# clue = Clue("", "", "")
# clue.convert_from_feature(
#     "explain clue: how are you explanation: it is an emotion",
#     "GOOD explanation: it is emotion",
#     "BAD explanation: oh well",
# )
# print(clue.to_map())
