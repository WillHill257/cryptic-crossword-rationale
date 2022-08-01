"""
sqlite> SELECT clue, answer, definition, annotation FROM clues LIMIT 2;
Acquisitive chap, as we see it (8)|COVETOUS|Acquisitive|COVE(chap) TO US (as we see it)
Back yard fencing weak and sagging (6)|DROOPY|sagging|reversal of all of the wordplay - YD(yard) containing POOR(weak)
"""

"""
sqlite> .schema clues
CREATE TABLE IF NOT EXISTS "clues" (
    clue TEXT,
    answer TEXT,
    definition TEXT,
    annotation TEXT,
    clue_number TEXT,
    puzzle_date TEXT,
    puzzle_name TEXT,
    source_url TEXT NOT NULL,
    source TEXT
);
CREATE INDEX clues_source_index ON clues ("source");
"""

import random
import sqlite3
import hashlib
from typing import *
from typing import List, Tuple, Dict
from clue import Clue
import copy
import logging

# Create and configure logger
# Setting the threshold of logger to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Creating an object
logger = logging.getLogger(__name__)

# helper function to query the DB 
def query_data(cursor: sqlite3.Cursor) -> List[Clue]:
  # build the SQL query
  SQL = """
    SELECT 
      REPLACE(clue, '"', ''''), 
      REPLACE(answer, '"', ''''), 
      REPLACE(annotation, '"', '''')
    FROM clues 
    WHERE 
      clue IS NOT NULL 
      AND clue != ""
      AND answer IS NOT NULL
      AND answer != ""
    ;  
  """

  # execute the query and get the result
  # result is a "2D array" -> rows are the records; cols are the attributes
  cursor.execute(SQL)
  
  # loop through the result, creating Clue objects
  data = []
  for row in cursor:
    # split the tuple into its constituent parts
    clue = row[0]
    answer = row[1]
    annotation = row[2] or ""

    # create the object and add to the list
    data.append(Clue(clue, answer, annotation))

  return data


# ----------------------------------------- RANDOM SPLIT ------------------------------------------

def make_random_split(all_clues: List[Clue], seed=42) -> Tuple[List[Clue]]:
  # randomly shuffle the data
  rng = random.Random(seed)
  rng.shuffle(all_clues)

  # splice the splits out
  val_start = int(0.6 * len(all_clues))
  test_start = int(0.8 * len(all_clues))
  train = all_clues[:val_start]
  val = all_clues[val_start:test_start]
  test = all_clues[test_start:]

  assert(len(all_clues) == len(train) + len(val) + len(test))

  logger.info("Random Split: Train Size      = {} of {}".format(len(train), len(all_clues)))
  logger.info("Random Split: Validation Size = {} of {}".format(len(val), len(all_clues)))
  logger.info("Random Split: Test Size       = {} of {}".format(len(test), len(all_clues)))
  
  return train, val, test

# ---------------------------------- WORD-INITIAL DISJOINT SPLIT ----------------------------------

# normal hash function is not deterministic across python runs
# adapted from: https://github.com/jsrozner/decrypt/blob/main/decrypt/scrape_parse/util.py
def hash(input: str):
  hash_obj = hashlib.md5(input.encode())
  return hash_obj.hexdigest()

def safe_hash(input: str) -> int:
  hex_hash = hash(input)
  return int(hex_hash, 16)

def make_disjoint_split(all_clues: List[Clue], word_initial_split: bool, seed=42) -> Tuple[List[Clue]]:

  # loop through each clue, computing the hash of the first 2 letters of the answer and spliting based on that
  train, val, test = [], [], []
  for clueObj in all_clues:
    # calculate the hash - all answers with the same (initial) letters will have the same hash and will belong to the same split
    # take modulus to get the correct split sizes (later)
    value_to_hash = clueObj.answer[:2] if word_initial_split else clueObj.answer

    h = safe_hash(value_to_hash) % 5  # normal hash function is not deterministic across python runs

    # hash of 0, 1, 2 -> train
    # hash of 3 -> validation
    # hash of 4 -> test
    # (i.e. a 60/20/20 split)
    if h < 3:
      train.append(clueObj)
    elif h < 4:
      val.append(clueObj)
    else:
      test.append(clueObj)

  # randomly shuffle the data in each split
  rng = random.Random(seed)
  rng.shuffle(train)
  rng.shuffle(val)
  rng.shuffle(test)

  assert(len(all_clues) == len(train) + len(val) + len(test))

  logger.info("{} Disjoint Split: Train Size      = {} of {}".format("Word-Initial" if word_initial_split else "Naive", len(train), len(all_clues)))
  logger.info("{} Disjoint Split: Validation Size = {} of {}".format("Word-Initial" if word_initial_split else "Naive", len(val), len(all_clues)))
  logger.info("{} Disjoint Split: Test Size       = {} of {}".format("Word-Initial" if word_initial_split else "Naive", len(test), len(all_clues)))
  
  return train, val, test

# ----------------------------------------- WRITE TO FILE -----------------------------------------

def write_splits_to_json(filename: str, train: List[Clue], validation: List[Clue], test: List[Clue]) -> None:
  # create the file (overwrite if it exists)
  filename_noext = "json/" + filename 
  f = open(filename_noext + ".json", "w")
  f.write('{')

  # add the data
  splits = [["train", train], ["validation", validation], ["test", test]]

  for i, split in enumerate(splits):
    # add the "title"
    f.write('"' + split[0] + '": [')

    # add the content
    for j, clueObj in enumerate(split[1]):
      f.write(clueObj.to_json())
      if (j != len(split[1]) - 1):
        f.write(", ")

    f.write(']')
    if (i != len(splits) - 1):
      f.write(', ')

  f.write('}')
  f.close()


# --------------------------------------------- MAIN ----------------------------------------------
if __name__ == "__main__":
  # connect to the DB
  con = sqlite3.connect("./data-annotated.sqlite3")

  # Once a Connection has been established, create a Cursor object and call its execute() method to perform SQL commands
  cur = con.cursor()

  # query the data from the DB
  logger.info("Querying data from the DB")
  data = query_data(cur)
  logger.info("Queried {} items".format(len(data)))

  # close the DB connection
  con.close()

  # create the data splits
  logger.info("Creating random split")
  train, validation, test = make_random_split(copy.deepcopy(data), 41)
  write_splits_to_json("random-split", train, validation, test)

  # naive disjoint split
  logger.info("Creating naive disjoint split")
  train, validation, test = make_disjoint_split(data, False, 43)
  write_splits_to_json("naive-disjoint-split", train, validation, test)

  # word-initial disjoint split
  logger.info("Creating word-initial disjoint split")
  train, validation, test = make_disjoint_split(data, True, 47)
  write_splits_to_json("word-initial-disjoint-split", train, validation, test)
  
