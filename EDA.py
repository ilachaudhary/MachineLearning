import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("house-prices/train.csv")

numerical_cols = df.describe().columns

#pearson correlation
df[numerical_cols].corr()

#Heatmap for correlation between numerical columns (Target: Sale Price)
plt.figure(figsize=(5,8))
sns.heatmap(
    df[numerical_cols].corr().sort_values("SalePrice", ascending=False)[["SalePrice"]],
    annot=True)
#plt.show()

plt.figure(figsize=(8,8))
sns.heatmap(
    df[numerical_cols].corr().iloc[:10,:10],
    annot=True)

plt.figure(figsize=(8,8))
sns.heatmap(
    df[numerical_cols].corr(),
    annot=False)
#plt.show()

#Relationship between categorical feature and Target
sns.displot(data=df, x="SalePrice", hue="KitchenQual", kind="kde")
#plt.show()

#Seaborne regplot to show Confidence Intervals
sns.regplot(df, x="GrLivArea", y="SalePrice", line_kws=dict(color="k"))
#plt.show()

#Box Plots
plt.figure(figsize=(3,8))
sns.boxplot(df, y="GrLivArea")
#plt.show()



