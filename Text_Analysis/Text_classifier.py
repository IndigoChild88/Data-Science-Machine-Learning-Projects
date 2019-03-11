# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:00:57 2019

@author: Albert Nunez
"""

import os
import io 
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirname, filename in os.walk(path):
        path = os.path.join(root, filename)
        
        inBody = False
        lines = []
        f = io.open(path, 'r', encoding = 'latin1')
        for line in f:
            if inBody:
                lines.append(line)    
            elif line == '\n':
                inBody = True
        f.close()
        message = '\n'.join(lines)
        yield path, message
        
#Method for reading text files in a directory
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        #class is for classification
        rows.append({'message' : message, 'class' : classification})
        index.append(filename)
    
    return DataFrame(rows, index = index)
#Method for reading 
def dataFrameFromCSV(Cell, classification):
    rows = []
    index = []
    for x in range(len(Cell)):
        #class is for classification
        rows.append({'message' : Cell[x], 'class' : classification})
        index.append('comments')
        
    
    return DataFrame(rows, index = index)

NSFW = pd.io.parsers.read_csv('NSFW.csv')
Tinder = pd.io.parsers.read_csv('Tinder.csv')

print(NSFW.keys(), "\nLength of one cell", len(NSFW["comments"]))
print(Tinder.keys(), "\nLength of one cell", len(Tinder["comments"]))

data = DataFrame({'message' : [], 'class' : []})
data = data.append(dataFrameFromCSV(NSFW['comments'], 'sexual'))
data = data.append(dataFrameFromCSV(Tinder['comments'] , 'normal'))

print(data.head())

data.head()

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

classifier = MultinomialNB()
targets = data['class'].values
print(targets[555])
classifier.fit(counts, targets)

examples = ['Thats cool yeah that would be great', 'boobs']
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)
print(predictions)
