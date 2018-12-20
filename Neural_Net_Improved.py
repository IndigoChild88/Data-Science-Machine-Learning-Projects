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
        #self.Saved="Not working"
        try:
            F1=open('w1.txt')
            F2=open('w2.txt')
            self.W1=np.loadtxt('w1.txt')
            self.W2=np.loadtxt('w2.txt')
            self.W11=np.loadtxt('w1.txt')
            self.W22=np.loadtxt('w2.txt')
            print("The wieght of W1: ",self.W1,"---------")
            print("The wieght of W2: ",self.W2,"---------")
            print("\nW2 shape: ",self.W2.shape())
            #self.Saved22=self.FileWeights(self.W1,self.W2)
        except:
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
    
    def sigmoidPrime(self, s):
        #derivative of sigmoid
        return s * (1 - s)
    
    def backward(self, X, y, o):
        # backward propagate through the network
        self.o_error = y-o # error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applyijng derivative of sigmoid to z2 error
        
        self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set ( hidden --> output) weights
        
    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
    def predict(self,Predict):
        print("\nPredicted data based on trainded weights: ")
        print("Input (scaled): \n", str(Predict))
        z=self.forward(Predict)
        print("Output: \n",str(z*100),"%")     
        return z
    def saveWeights(self):
        np.savetxt("w1.txt", self.W1, fmt ="%s")
        np.savetxt("w2.txt", self.W2, fmt ="%s")
    def FileWeights(self, w1, w2):
        v = "FUCK"
        #"This is W1 from file: "+self.w1+"\nThis is w2 from file: "+ self.w2
        return v
    def ReturnV(self):
        h = self.W22
        return self.W11,h
    def Accuracy(self, result, actual):
        accur=0
        if result/actual > 1:
            accur = 2-(result/actual)
        else:
            accur = result/actual
        return accur *100
NN = Neural_Network()

#defining our output
o = NN.forward(X)
print ("Predicted Output: \n ", str(o))
print ("Actual Output: \n", str(y))

# training code for iterations
for i in range(1000): # trains the NN 1,000 times
    print ("Input: \n", str(X))
    print ("Actual output: \n", str(y))
    print ("Predicted Output: \n", str(NN.forward(X)))
    print ("Loss: \n", str(np.mean(np.square(y - NN.forward(X))))) # mean sum square loss
    print ("Input: \n", str(y))
    print ("\n")
    NN.train(X, y)

print ("Predicted Output: \n ", str(o))
print ("Actual Output: \n", str(y))

xPredicted = np.array(([4, 9]), dtype=float)
#scale this
xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted(input data for prediction)
Actual=85
Result =NN.predict(xPredicted)
print("this is the Result, ",Result[0])
Error = 100 - ((Result[0]*100)-Actual)
print ("Accuracy: ",NN.Accuracy(Result[0]*100,Actual), "%")
NN.saveWeights()
print("This is Saved: ",NN.ReturnV())