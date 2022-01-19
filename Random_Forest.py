from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split

path = Path("data/houses.csv")
df = pd.read_csv(path)

#Selecting variables
columns = ['SalePrice','LotArea', 'YearBuilt', 'YrSold', 'OverallQual', 'OverallCond', 'YearRemodAdd', '1stFlrSF', ]
X=df[columns]  # Features
y=df['CentralAir']  # Labels

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

#Random forest model
clf=RandomForestClassifier(random_state=1)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

#Measuring quality
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('pres', tp/(fp+tp))
print('sensiviteit', tp/(fn+tp))
print('specifity', tn/(fp+tn))

#Confusion matrix plot
plot_confusion_matrix(clf, X_test, y_test)
# plt.show()

#Plot impact X on Y
print(classification_report(y_test, y_pred))
feature_imp = pd.Series(clf.feature_importances_, index=columns).sort_values(ascending=False)
feature_imp

sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
# plt.show()