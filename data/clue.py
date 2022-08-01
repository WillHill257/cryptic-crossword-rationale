# create a clue class

from typing import *

class Clue:
  def __init__(self, clue: str, answer: str, annotation: str, predicted_rationale: str = "") -> None:    
    self.clue = clue
    self.answer = answer
    self.annotation = annotation
    self.predicted_rationale = predicted_rationale

  def to_json(self) -> str:
    return '{"clue": "' + self.clue + '", "answer": "' + self.answer + '", "annotation": "' + self.annotation + '", "rationale": "' + self.predicted_rationale + '"}'
