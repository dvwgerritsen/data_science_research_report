from pathlib import Path

import numpy as np
import pandas as pd

# File path
path = Path("../../data/raw/us_accidents.csv")

# Reads CSV file and puts it into pandas data frame
data = pd.read_csv(path)

columnNames = ""
for col in data.columns:
    columnNames += str(col) + " "

print(columnNames)
