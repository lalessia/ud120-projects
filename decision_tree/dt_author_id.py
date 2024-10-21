#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Initialize the Decision Tree Classifier with min_samples_split=40
clf = DecisionTreeClassifier(min_samples_split=40)


# Train the model and measure the time taken
t0 = time()
clf.fit(features_train, labels_train)
print(f"Training time: {round(time()-t0, 3)}s")

# Predict on the test set and measure the time taken
t1 = time()
pred = clf.predict(features_test)
print(f"Prediction time: {round(time()-t1, 3)}s")

# Calculate accuracy of the model
accuracy = accuracy_score(labels_test, pred)
print(f"Accuracy: {accuracy}")

#########################################################


