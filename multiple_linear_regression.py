from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
import numpy as np

warnings.filterwarnings("ignore")

path = Path("data/houses.csv")

df = pd.read_csv(path)

mlrdf = df
mlrdf.corrwith(mlrdf['SalePrice'])

# Assign X and Y axis
x1 = mlrdf['GrLivArea']
x2 = mlrdf['GarageArea']
X = mlrdf[['GarageArea', 'GrLivArea']]
y = mlrdf['SalePrice']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(r2_score(y_test, y_pred))

maxGrLivArea = 4000
maxGarageArea = 1100

y_predLineStart = [[0, 0]]
y_predLineEnd = [[4000, 1100]]
yStart = regressor.predict(y_predLineStart)
yEnd = regressor.predict(y_predLineEnd)

x1Plot = [0, 4000]
x2Plot = [0, 1100]
yPlot = np.array([yStart, yEnd]).flatten().tolist()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, c='blue', marker='o')
ax.plot3D(x1Plot, x2Plot, yPlot, color='red', linewidth=5)
# set your labels
ax.set_xlabel('GarageArea')
ax.set_ylabel('GrLivArea')
ax.set_zlabel('Price')
plt.show()
