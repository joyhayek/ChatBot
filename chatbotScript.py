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
        # Add the tokens to the docs_x list
        docs_x.append(tokens)
        # Add the tag of each pattern to the docs_y list
        docs_y.append(intent["tag"])
        
    # Add the tags to the tags list if not in list aleady
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        
# Stem all tokens
words = [stemmer.stem(token.lower()) for token in words if token != "?"]
# Store the words in a sorted list that doesn't contain any duplicates
words = sorted(list(set(words)))
# Sort the labels/tags
labels = sorted(labels)

# Perform one-hot-encoding on the words - neural networks only understand numbers and not words
# We will store 0 if the word doesn't exist in the prompt, and 1 if it does
training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    # Stem each token in docs_x
    stemmed_tokens = [stemmer.stem(d) for d in doc]
    # Encode each stemmed word based on the pattern and append to bag
    for word in words:
        if word in stemmed_tokens:
            bag.append(1)
        else:
            bag.append(0)

    # Generate the encoded output lists
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)
    
# Turn the training and output lists into numpy arrays
training = np.array(training)
output = np.array(output)