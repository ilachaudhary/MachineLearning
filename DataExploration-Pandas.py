import pandas as pd

df = pd.read_csv("house-prices/train.csv")

#Initial data exploration
df.head()

#Size of data
df.shape

#Data types and nulls
df.info()

#Summary stats of data
df.describe()

#Summary of categorical features
df["PoolQC"].value_counts()
df["GarageQual"].value_counts()