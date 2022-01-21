from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

path = Path("data/houses.csv")
df = pd.read_csv(path)

df['Fence'].replace(['GdPrv', 'MnPrv', 'GdWo', 'MnWw'], 'F', regex=True, inplace=True)
# df['Street'].replace(['Grvl'], 0, regex=True, inplace=True)
# df['Street'].replace(['Pave'], 1, regex=True, inplace=True)
print(df['Street'])
df['Fence'].replace(['NA'], 'NF', regex=True, inplace=True)

df = df[df['Alley'].notna()]

x1 = df['OverallQual']
x2 = df['OverallCond']
X = df[['OverallQual', 'OverallCond']]
y = df['Alley']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

sns.scatterplot(x=x1, y=x2, hue=y, palette="pastel", s=20)

from sklearn import svm, metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

clf = svm.SVC(kernel='linear', random_state=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

support_vector_indices = clf.support_
print(clf.support_)
# Get support vectors themselves
support_vectors = clf.support_vectors_

# Visualize support vectors
plt.scatter(support_vectors, support_vectors, color='red', marker=',', s=1)
plt.title('Linearly separable data with support vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

