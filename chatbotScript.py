# Start by importing modules that will be used to read in the data from the json file

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy 
import tflearn
import tensorflow
import json
import random
import pickle

stemmer = LancasterStemmer()

# Open the json file and load it
with open("data.json") as json_data:
    data = json.load(json_data)

try:
    # Load the data from the pickle file
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    
except:
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
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            # Add the tokens to the docs_x list
            docs_x.append(wrds)
            # Add the tag of each pattern to the docs_y list
            docs_y.append(intent["tag"])
            
        # Add the tags to the tags list if not in list aleady
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
            
    # Stem all tokens
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
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
        wrds = [stemmer.stem(w.lower()) for w in doc]
        # Encode each stemmed word based on the pattern and append to bag
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        # Generate the encoded output lists
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)
        
    # Turn the training and output lists into numpy arrays
    training = numpy.array(training)
    output = numpy.array(output)
    
    # Save all of this data in a pickle file
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Creating the model
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

# Training the model
model = tflearn.DNN(net)

# Fit the model
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
# Save the model
model.save("model.tflearn")


# Start making predictions
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return numpy.array(bag)

def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break 
        
        results = model.predict([bag_of_words(inp, words)])[0]
        # Returns the index of the highest probability
        results_index = numpy.argmax(results)
        # Check the value of the probability
        
        # If valid, print appropriate response
        if results[results_index] > 0.7:
            # Get the corresponding response from the tag from the json file
            # Get the corresponding tag
            tag = labels[results_index]
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
            # Display the responses
            print(random.choice(responses))       
            
        # If not valid, print a message asking user to try again
        else:
            print("I'm not sure I understand. Please try again or ask another question") 
            
chat()