from pathlib import Path
import pandas as pd

# Path of file
from matplotlib import pyplot as plt

path = Path("../../data/processed/marketing_campaign.csv")

# Reads CSV file and puts it into pandas data frame
df = pd.read_csv(path, sep='\t', engine='python')

#Shows plot
plt.scatter(df.get("Income"), df.get("MntWines"))
plt.show()