# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 15:07:25 2018

@author: acn00
"""
import numpy as np
import pandas as pd
import mglearn
from numpy import array
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
#from sklearn import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import os,sys
from skimage.io import imread
clf=svm.SVC(gamma=0.001, C=100)
import tensorflow as tf

n_inputs =3#3d inputs
n_hidden = 3 #2D codings
n_outputs = n_inputs

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape =[None,None,None, n_inputs])
hidden = tf.layers.dense(X,n_hidden)
outputs = tf.layers.dense(hidden, n_outputs)

reconstruction_loss = tf.reduce_mean(tf.square(outputs-X)) #MSE

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op= optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()

# =============================================================================
#upload dataset
folder ="flowers\sunflower"
bigfile="flowers\sunflower\\"
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
I=[]
V=0
N=[]
for x in files:
    #print(x)
   # img=imread(bigfile+x)
    img=Image.open(bigfile+x)
    #I.append([array(img)])
    imageR=img.resize((200,200),Image.NEAREST)
    img.shape=(500,300)
    #imageR.save("TEST.png")
    #imageROW=imageR.flatten()
    I.append(np.asarray(imageR))
    N.append([1])
    V +=1

I=np.asarray(I,dtype=float)
X_train, X_test=I[0:300],I[0:10]
#X_train=array(X_train)
print(I[1], I.shape)

# =============================================================================
n_iterations=100
codings = hidden

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X: X_train}) # no labels (unsupervised)
    codings_val = codings.eval(feed_dict = {X: X_test})
codings_val=array(codings_val,dtype=float)
print(len(codings_val[0]),type(codings_val),codings_val[1])
#svimg=im.fromarray(data.astype('uint8'))
PP= Image.fromarray(codings_val[1].astype('uint8'))
print(PP)
PP.save("AutoPic.png")
Check_pic=Image.fromarray(X_test[1].astype('uint8'))
print(Check_pic)
Check_pic.save("Check_pic.png")
############################################################ SECOND TEST
PP2= Image.fromarray(codings_val[2].astype('uint8'))
print(PP2)
PP2.save("AutoPic2.png")
Check_pic2=Image.fromarray(X_test[2].astype('uint8'))
print(Check_pic2)
Check_pic2.save("Check_pic2.png")