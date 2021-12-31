from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
import numpy as np

path = Path("data/houses.csv")
df = pd.read_csv(path)

##Convert Y/N into 1/0
#df.CentralAir.replace(('Y', 'N'), (1, 0), inplace=True)
#print(df.CentralAir)

##Showing distrubution
#print(df['CentralAir'].value_counts())
#sns.countplot(x='CentralAir' ,data=df, palette='hls')
#plt.show()
#plt.savefig('countplot')

##percentage CentralAIr
#count_no_sub = len(df[df['CentralAir']==0])
#count_sub = len(df[df['CentralAir']==1])
#pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
#print("percentage of no CentralAir is", pct_of_no_sub*100)
#pct_of_sub = count_sub/(count_no_sub+count_sub)
#print("percentage of CentralAir", pct_of_sub*100)

#Select columns make predictor
columns = ['LotArea']
x = df[columns]
y = df.CentralAir

#split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(x_train,y_train)

#
y_pred=logreg.predict(x_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
#plt.show()

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

