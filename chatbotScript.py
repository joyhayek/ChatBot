# Start by importing modules that will be used to read in the data from the json file

import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json
import random

stemmer = LancasterStemmer()

# Open the json file and load it
with open("data.json") as json_data:
    data = json.load(json_data)

# print(data)

# Create lists to store the data in the json file
words = []
# This stores the tags
labels = []
# this stores the patterns
docs_x = []
# this stores the tag of each pattern
docs_y = []

# Loop through the data 
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize each word in patterns and add the tokens to the words list
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        # Add each pattern to the docs_x list
        docs_x.append(pattern)
        # Add the tag of each pattern to the docs_y list
        docs_y.append(intent["tag"])
        
    # Add the tags to the tags list if not in list aleady
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        

