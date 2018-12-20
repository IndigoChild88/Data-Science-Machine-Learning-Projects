# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 00:32:43 2018

@author: acn00
"""

import numpy as np
#X = [hours sleeping, hour,s studying], y = score on a test
X = np.array (([4,9],[8,8],[2,4],[9,2],[5,5]), dtype=float)
y = np.array(([85], [93], [65],[51],[82]),dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max tes scors is 100
print(X)

class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        
        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (hidden x input)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (hidden x output)
        
    def forward(self, X):
        #foward propagation through the network
        self.z = np.dot(X, self.W1)# dot product of X(input) first set of hidden x input wieghts
        self.z2 = self.sigmoid(self.z) # activation fiunction
        self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layr z2 and second hidden to output
        o = self.sigmoid(self.z3)# final activation
        return o
        
    def sigmoid(self,s):
        # activation function
        return 1/(1+np.exp(-s))

NN = Neural_Network()

#defining our output
o = NN.forward(X)

print ("Predicted Output: \n ", str(o))
print ("Actual Output: \n", str(y))
