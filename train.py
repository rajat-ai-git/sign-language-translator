import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

import csv
df = pd.read_csv('data.csv', header=None)
print(df)
x = df.drop(0, axis=1)
y = df[0]
print(x.shape)
print()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

print(X_train.shape)
print(y_train.shape)
model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy: ", accuracy)


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model Saved!")
