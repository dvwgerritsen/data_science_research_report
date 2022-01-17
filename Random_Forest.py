from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

path = Path("data/houses.csv")
df = pd.read_csv(path)

#print(df.columns.tolist())
#print(df.)
columns = ['SalePrice','LotArea', 'YearBuilt', 'YrSold', 'OverallQual', 'OverallCond', 'YearRemodAdd', '1stFlrSF', ]
X=df[columns]  # Features
y=df['LotShape']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

feature_imp = pd.Series(clf.feature_importances_,index=columns).sort_values(ascending=False)
feature_imp

sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()