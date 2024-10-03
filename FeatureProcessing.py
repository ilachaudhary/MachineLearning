import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import matplotlib.pyplot as plt

df = pd.read_csv("house-prices/train.csv")

#One-hot Encoding
pd.get_dummies(df["Neighborhood"], dtype=int)
pd.concat(
  [
   pd.get_dummies(df["Neighborhood"], prefix="Neighborhood", dtype=int),
   df
  ],
  axis=1
 )

#Ordinal Mapping and adding a new column
ordinal_mapping = {
 "Po":	0,
 "Fa":	1,
 "TA": 2,
 "Gd": 3,
 "Ex": 4,
}
df["KitchenQual_ord"] = df.apply(lambda x: ordinal_mapping[x["KitchenQual"]], axis=1)

#Feature Scaling
np.log(df["LotArea"])
q = QuantileTransformer()
q.fit_transform(df[["LotArea"]])

#Standard Scaler
s = StandardScaler()
s.fit_transform(df[["LotArea"]])

#Min Max Scaler
m = MinMaxScaler()
m.fit_transform(df[["LotArea"]])

#Generate New Colums with Polynomial features
df["OverallQual^2"] = df["OverallQual"]**2

df.fillna({"PoolQC": "NA", "LotFrontage": 0})
numerical_cols = df.describe().columns
mean_values = df[numerical_cols].mean().to_dict()
df.fillna(mean_values)

#Feature Selection
numerical_cols = numerical_cols.drop("SalePrice")

#Generate 10-D object with PCA
pca = PCA(n_components = 10, svd_solver = "full")
pca.fit_transform(df[numerical_cols].fillna(0))

#Create DataFrame out of Numpy array
pd.DataFrame(pca.fit_transform(df[numerical_cols].fillna(0)))

#Variance through pca
pca = PCA(svd_solver='full')
df_pca = pd.DataFrame(pca.fit_transform(df[numerical_cols].fillna(0)))
df_pca.var()

#Using first scale before PCA
s = StandardScaler()
df_scaled = s.fit_transform(df[numerical_cols].fillna(0))

pca = PCA(svd_solver='full')
df_pca = pd.DataFrame(pca.fit_transform(df_scaled))
variances = df_pca.var()

#Data visualization
plt.plot(variances)
plt.show()