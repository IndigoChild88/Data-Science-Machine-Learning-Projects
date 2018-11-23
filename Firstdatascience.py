# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 12:42:27 2018

@author: acn00
"""
import numpy as np
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris_dataset= load_iris()
#from sklearn import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

print("Keys of iris data set:  \n{}".format(iris_dataset.keys()))


#Shows the description of the dataset
print("DESCRIPTION OF DATA SET:  \n")
print(iris_dataset['DESCR'][:192]+"\n...")

#Shows the taraget values
print("Target Names: {}".format(iris_dataset['target_names']))

#Feature name is descriiption of each feature
print("Name of the features: \n{}".format(iris_dataset['feature_names']))

#Shows the type data 
#print("Type of data: {}".format(iris_dataset['data']))
print("\n")
#Print the shape of the data 
print("Shape of data: {}".format(iris_dataset['data'].shape))

#Print the first seven rows of the data in the data set
print("First five rows of data:\n{}".format(iris_dataset['data'][:7]))
print()
# print the data for the target
print("First seven rows for the target:  \n{}".format(iris_dataset['target'][:7]))

#Sets up training data for model
X_train, X_test, y_train, y_test= train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)

#Creates a data plot 
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

#Configuration for the ploting
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins':20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)
                 
#Train the model
knn.fit(X_train, y_train)
#create a new test variable
X_new=np.array([[5,2.9, 1,0.2]])
print("X_new.shape{} ".format(X_new.shape))

#Shows the predicted value
prediction=knn.predict(X_new)
print("The predicted value is: {}".format(prediction))
print("The Prediction target name: {}".format(iris_dataset['target_names'][prediction]))
#Shows how the prediction of the data set should be in a certain class of the dataset
print("The value {}".format(prediction)+" Means it's in the class of {}".format(iris_dataset['target_names'][prediction]))



#Evaluating the MODEL
y_pred = knn.predict(X_test)
#Help compare the predicted values to the actual values which creates the accuracy percentage
print("y_pred: {}".format(y_pred))
print("y_test: {}".format((y_test)))
print("\n This array checks which values match: {}".format((y_pred==y_test)))

#print("\n x_test: {:.2f}".format(np.mean(y_pred)))
#print("\n y_test: {:.2f}".format(np.mean(y_test)))

# This shows the accuracy of the model
print("\n Test set score: {:.2f}".format(np.mean(y_pred==y_test)))

#print("\n Test set score: {:.2f}".format(knn.score(X_test, y_test)))
#print("Test set score: {:.2f}".format(knn.score(y_test, X_test)))

