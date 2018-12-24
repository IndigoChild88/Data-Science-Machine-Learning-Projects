# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 02:18:18 2018

@author: Albert Nunez
"""

from sklearn.tree import DecisionTreeRegressor
import pandas as pd

#Model uses a decision tree regressor from sklearn
kansas_model= DecisionTreeRegressor()

#loads house data
house_data=pd.read_csv("kc_house_data.csv")

#Prints out the feature names
print(house_data.columns)

#Selects specific features
house_feature=['sqft_lot15','sqft_living15','bedrooms','waterfront',
               'bathrooms',
               'floors']

#X now contains house features
X=house_data[house_feature]

#y uses the price columns
y=house_data.price

#Trains the model
kansas_model.fit(X,y)

#Ask the model to predict the price
Predict = kansas_model.predict(X[:10])

#Print out the prediction 
print("This is the prediction: ",Predict)

#Print out the actual prices
print("Actual price of the houses: ", y[:10])

#print out the differences between the prediction and actual price
print("Error: ", Predict-y[:10])

