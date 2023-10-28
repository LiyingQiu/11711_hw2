#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:18:46 2023

@author: liyingqiu

This file tokenize the data from text to tokens and export as a txt file.
"""

import os
import spacy
import re
from extract_sentence_pdf import extract_sentence

def tokenize_data(pdf_path):

    content = extract_sentence(pdf_path)

    content = re.sub(r'-\n', '', content)

    nlp = spacy.load("en_core_web_sm")

    doc = nlp(content)

    tokens = [token.text for token in doc]

    return tokens

# separate by sentence
# remove "- " and connect the words
# remove "\n", ",", ".", "(", ")", ":", ";", "!", "?", "``", "''", "’", "“", "”", "–", "—", "‘", "•", "…", ", "- "
def clean_data(tokens):
    # add "\n" to the end of each sentence, which ends with ".", "!", "?"
    tokens = [token + "\n" if token in [".", "!", "?"] else token for token in tokens]

    # remove "- " and connect the words
    # tokens = [token.replace("- ", "") if token.endswith("-") else token for token in tokens]

    # remove "\n", ",", ".", "(", ")", ":", ";", "!", "?", "``", "''", "’", "“", "”", "–", "—", "‘", "•", "…", "
    tokens = [token for token in tokens if token not in ["\n", ",", "(", ")", ":", ";", "``", "''", "’", "“", "”", "–", "—", "‘", "•", "…", " ",  "- ", "- "]]

    return tokens

pdf_path = "./Data/2023.acl-long.114.pdf"
tokens = tokenize_data(pdf_path)

# tokenize the data
tokens = clean_data(tokens)

# export as txt file
with open("./Token/2023.acl-long.114.txt", "w") as f:
    for token in tokens:
        f.write(token + " ")





