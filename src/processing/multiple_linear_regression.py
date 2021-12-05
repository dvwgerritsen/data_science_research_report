from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# Path of file
path = Path("../../data/processed/marketing_campaign.csv")

# Reads CSV file and puts it into pandas data frame
df = pd.read_csv(path, sep='\t', engine='python')

# Shows plot
x = np.array(list(df.get("Income"))).reshape((-1, 1))
y = np.array(list(df.get("MntWines"))).reshape((-1, 1))
plt.scatter(x, y)
plt.show()
model = LinearRegression().fit(x, y)
xfit = np.linspace(10000, 120000, 50)
yfit = model.predict(xfit[:, np.newaxis])
plt.scatter(x, y)
plt.plot(xfit, yfit)
plt.show()
