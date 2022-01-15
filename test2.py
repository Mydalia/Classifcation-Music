import pandas as pd 
from sklearn.model_selection import train_test_split

tracks = pd.read_excel("./Ressources/spotify_genre_final.xlsx")

print(tracks.head())
print(tracks.columns)

y = tracks['Genre'].values
X = tracks.drop('Genre', axis=1).values

print(y)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)