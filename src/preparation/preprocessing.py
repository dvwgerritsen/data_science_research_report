from pathlib import Path

import numpy as np
import pandas as pd

# Path of file
path = Path("../../data/raw/marketing_campaign.csv")

# Reads CSV file and puts it into pandas data frame
df = pd.read_csv(path, sep='\t', engine='python')
# Filters get applied to dataframe
fdf = df.where(df.get('Income') < 150000).dropna()


# Converts values that were automatically converted to floats to integers if possible
def convert(x):
    try:
        return x.astype(int)
    except:
        return x


# Generates processed datafile
fdf.apply(convert).to_csv(path_or_buf="../../data/processed/marketing_campaign.csv", sep="\t", index=0)
