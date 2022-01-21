from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

path = Path("data/houses.csv")
df = pd.read_csv(path)


columns = ['1stFlrSF', '2ndFlrSF', 'OverallCond']
X = df[['1stFlrSF', '2ndFlrSF', 'OverallCond']]
y = df['HouseStyle']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(random_state=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))
