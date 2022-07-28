# create a clue class

from typing import *

class Clue:
  def __init__(self, clue: str, answer: str, annotation: str) -> None:    
    self.clue = clue
    self.answer = answer
    self.annotation = annotation

  def to_string(self) -> str:
    return "clue: " + self.clue + "; answer: " + self.answer + "; annotation: " + self.annotation

  def to_json(self) -> str:
    return '{"clue": "' + self.clue + '", "answer": "' + self.answer + '", "annotation": "' + self.annotation + '"}'
