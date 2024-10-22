import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# EDA and feature preprocessing
df = pd.read_csv("loan_approval_dataset.csv")
print(df.head(10))

df.columns = [x.strip(' ') for x in df.columns]
numerical_cols = [x.strip(' ') for x in df.describe().columns.drop("loan_id")]

ordinal_mapping = {
    " Approved": 1,
    " Rejected": 0,
}
df['loan_mapped'] = df['loan_status'].map(ordinal_mapping)
correlation_matrix = df[['no_of_dependents', 'income_annum', 'loan_amount',
                         'loan_term', 'cibil_score', 'residential_assets_value',
                         'commercial_assets_value', 'luxury_assets_value',
                         'bank_asset_value', 'loan_mapped']].corr()

# Data visualization
# Pearson Correlation
sns.heatmap(correlation_matrix.sort_values("loan_mapped", ascending=False)[["loan_mapped"]],
            cmap='Greens',
            annot=True)
plt.show()

# Heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix.iloc[:10, :10],
            cmap='Oranges',
            annot=True)
plt.show()


# Check skewness of loan_status
freq = sns.countplot(x='loan_status', data=df, palette=['#FFB6C1', '#B0E0E6'], hue='loan_status', width=0.4)
freq.set_xlabel('Loan Status')
freq.set_ylabel('Count')
plt.title('Count of Loan Status')
plt.show()