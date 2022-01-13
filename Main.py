import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import sklearn
from IPython.display import display
from sklearn.utils import Bunch

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

mfd = load_spotify_dataset()

X = mfd.data

y = mfd.target

print(X)
print(y)