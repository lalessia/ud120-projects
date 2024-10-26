#!/usr/bin/python3

import random
import numpy
import matplotlib.pyplot as plt
import joblib

from outlier_cleaner import outlierCleaner


### load up some practice data with outliers in it
ages = joblib.load( open("./practice_outliers_ages.pkl", "rb") )
net_worths = joblib.load( open("./practice_outliers_net_worths.pkl", "rb") )



### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))
from sklearn.model_selection import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)

print("reg.coef_ with all data", reg.coef_)
print("reg.intercept_ with all data", reg.intercept_)

pred = reg.predict(ages_test)
print("reg.score with all data", reg.score(ages_test, net_worths_test))

try:
    plt.plot(ages, reg.predict(ages), color="blue")
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.show()

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner( predictions, ages_train, net_worths_train )
    
    
    if cleaned_data:  # assicurati che non sia vuoto
        ages_cleaned, net_worths_cleaned, errors = zip(*cleaned_data)
        
        # Trasforma in numpy array e ridimensiona
        ages_cleaned = numpy.reshape(numpy.array(ages_cleaned), (len(ages_cleaned), 1))
        net_worths_cleaned = numpy.reshape(numpy.array(net_worths_cleaned), (len(net_worths_cleaned), 1))

        # Dividi nuovamente i dati in train e test
        ages_train_cleaned, ages_test_cleaned, net_worths_train_cleaned, net_worths_test_cleaned = train_test_split(
            ages_cleaned, net_worths_cleaned, test_size=0.1, random_state=42
        )

        # Rifai la regressione sul nuovo training set
        reg2 = LinearRegression()
        reg2.fit(ages_train_cleaned, net_worths_train_cleaned)
        print("new slope:", reg2.coef_)
        print("new intercept without outlieri:i", reg2.intercept_)
        print("new score sul without outlier:", reg2.score(ages_test_cleaned, net_worths_test_cleaned))

except NameError:
    print("Your regression object doesn't exist, or isn't name reg")
    print("Can't make predictions to use in identifying outliers")






'''
TO RESTORE:
### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages       = numpy.reshape( numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape( numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        plt.plot(ages, reg.predict(ages), color="blue")
    except NameError:
        print("You don't seem to have regression imported/created,")
        print("   or else your regression object isn't named reg")
        print("   either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()


else:
    print("outlierCleaner() is returning an empty list, no refitting to be done")
    
    
'''