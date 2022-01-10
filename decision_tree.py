import graphviz
import pandas as pd
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt
from seaborn.external.six import StringIO
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.model_selection import train_test_split

import numpy as np

path = Path("data/houses.csv")
df = pd.read_csv(path)

#columns = ['ExterQual', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'PoolQC', 'Fence']
columns = ['YearBuilt', 'LotArea', 'SalePrice']
X = df[columns]
y = df['CentralAir']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Create Decision Tree classifer object
clf = DecisionTreeRegressor()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

target = list(y.unique())
feature_names = list(X.columns)
print(target)
print(feature_names)

from sklearn import tree

tree.export_graphviz(clf,
                     out_file="tree.dot",
                     feature_names = feature_names,
                     class_names=target,
                     filled = True)

from sklearn.tree import export_text
r = export_text(clf, feature_names=feature_names)
print(r)


#fig.savefig("decistion_tree.png")