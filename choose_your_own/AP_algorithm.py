#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score


def evaluate_model(clf, features_train, labels_train, features_test, labels_test, model_name):
    t0 = time()
    clf.fit(features_train, labels_train)
    print(f"{model_name} - Training time: {round(time()-t0, 3)}s")

    t1 = time()
    predictions = clf.predict(features_test)
    print(f"{model_name} - Prediction time: {round(time()-t1, 3)}s")

    accuracy = accuracy_score(labels_test, predictions)
    print(f"{model_name} - Accuracy: {accuracy}")
    
    try:
        prettyPicture(clf, features_test, labels_test)
    except NameError:
        pass
    return accuracy

# K-Nearest Neighbors
knn_clf = KNeighborsClassifier(n_neighbors=4)
evaluate_model(knn_clf, features_train, labels_train, features_test, labels_test, "KNN")

# AdaBoost
adaboost_clf = AdaBoostClassifier(n_estimators=50)
evaluate_model(adaboost_clf, features_train, labels_train, features_test, labels_test, "AdaBoost")

# Random Forest
random_forest_clf = RandomForestClassifier(n_estimators=100, min_samples_split=20)
evaluate_model(random_forest_clf, features_train, labels_train, features_test, labels_test, "Random Forest")







