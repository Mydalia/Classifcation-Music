import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import sklearn
from IPython.display import display
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_spotify_dataset():
    with open(r'./Ressources/spotify_genre_final.csv') as csv_file:
        data_reader = csv.reader(csv_file)
        feature_names = next(data_reader)[:-1]
        data = []
        target = []
        for row in data_reader:
            features = row[:-1]
            label = row[-1]
            data.append(features)
            target.append(label)
        
        data = np.array(data)
        target = np.array(target)
    return Bunch(data=data, target=target, feature_names=feature_names)

msd = load_spotify_dataset()

X = msd.data
y = msd.target

X_train, X_test, y_train, y_test = train_test_split(X[:,[2,8]], y, test_size=0.80)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
print(X)
print(y)

clf_lr = LogisticRegression(solver='lbfgs') # clf = classifier lr = logistic regression

clf_lr.fit(X_train, y_train) 

print("intercept: {}".format(clf_lr.intercept_))
print("weights:   {}".format(clf_lr.coef_))

t = X_test[:1,:] # build array only containing the first example from test using slicing
pred = clf_lr.predict(t) # predict() requires n-dimensional array
pred_pr = clf_lr.predict_proba(t) 

t_pred = pred[0]
t_pred_pr = pred_pr[0]
print("Prediction: {}, probability: {}".format(t_pred,t_pred_pr[t_pred]))

train_score = clf_lr.score(X_train,y_train)
test_score = clf_lr.score(X_test,y_test)

print("Training set score: {:.2f} ".format(train_score))
print("Test set score: {:.2f} ".format(test_score))