#!/usr/bin/env python3

import urllib
import requests
import csv

def enc(txt):
  return urllib.parse.quote(txt)

def coref(txt):
  r = requests.get("https://coref.huggingface.co/coref?text=" + enc(txt))
  return r.json()


with open("gap-validation.tsv") as tsvfile:
  reader = csv.DictReader(tsvfile, dialect='excel-tab')
  for row in reader:

    text = row["Text"]

    pronoun = row["Pronoun"]
    pronoun_offset = int(row["Pronoun-offset"])
    pronoun_idx = -1

    A_name = row["A"]
    A_offset = int(row["A-offset"])
    A_idx = -1

    B_name = row["B"]
    B_offset = int(row["B-offset"])
    B_idx = -1

    result = coref(text)

    if "mentions" in result:
      for mention in result["mentions"]:
        if mention["start"] <= pronoun_offset < mention["end"]:
          pronoun_idx = mention["index"]
        if mention["start"] <= A_offset < mention["end"]:
          A_idx = mention["index"]
        if mention["start"] <= B_offset < mention["end"]:
          B_idx = mention["index"]

    A_score = -1
    B_score = -1

    if pronoun_idx != -1 and A_idx != -1:
      if pronoun_idx > A_idx:
        A_score = result["pairScores"][str(pronoun_idx)][str(A_idx)]
      elif pronoun_idx < A_idx:
        A_score = result["pairScores"][str(A_idx)][str(pronoun_idx)]

    if pronoun_idx != -1 and B_idx != -1:
      if pronoun_idx > B_idx:
        B_score = result["pairScores"][str(pronoun_idx)][str(B_idx)]
      elif pronoun_idx < B_idx:
        B_score = result["pairScores"][str(B_idx)][str(pronoun_idx)]

    print(row["ID"], A_score, B_score)
