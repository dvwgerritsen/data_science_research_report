# from pathlib import Path
#
# import pandas as pd
#
# # Path of file
# path = Path("../../data/raw/marketing_campaign.csv")
#
# # Reads CSV file and puts it into pandas data frame
# df = pd.read_csv(path, sep='\t', engine='python')
# # Data gets filtered by dropping empty records and adding specific filters
# df = df.dropna()
# df = df.where(df["Income"] < 50000)
# df = df.where(df["MntWines"] < 200)
# df = df.where(df["Year_Birth"] > 1940)
# filterMarital_Status = df["Marital_Status"].isin(["Single", "Together", "Divorced", "Widow", "Married"])
#
#
# # Converts values that were automatically converted to floats to integers if possible
# def convert(x):
#     try:
#         return x.astype(int)
#     except:
#         return x
#
#
# # Generates processed datafile
# df[filterMarital_Status].to_csv(path_or_buf="../../data/processed/marketing_campaign.csv", sep="\t",
#                                 index=0)
