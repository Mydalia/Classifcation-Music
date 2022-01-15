import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

tracks = pd.read_excel("./Ressources/spotify_genre_final.xlsx")

print(tracks.head())
print(tracks.columns)

y = tracks['Genre'].values
X = tracks.drop('Genre', axis=1).values

print(y)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

clf_lr = LogisticRegression(solver='lbfgs')

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