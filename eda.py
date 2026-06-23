import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Set style
sns.set_style("whitegrid")

# ===========================
# 1. Churn Distribution
# ===========================
plt.figure(figsize=(6,4))
sns.countplot(x='Exited', data=df)
plt.title('Customer Churn Distribution')
plt.savefig('churn_distribution.png')
plt.show()

# ===========================
# 2. Gender-wise Churn
# ===========================
plt.figure(figsize=(6,4))
sns.countplot(x='Gender', hue='Exited', data=df)
plt.title('Gender-wise Customer Churn')
plt.savefig('gender_churn.png')
plt.show()

# ===========================
# 3. Geography-wise Churn
# ===========================
plt.figure(figsize=(8,5))
sns.countplot(x='Geography', hue='Exited', data=df)
plt.title('Geography-wise Customer Churn')
plt.savefig('geography_churn.png')
plt.show()

# ===========================
# 4. Age Distribution
# ===========================
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.savefig('age_distribution.png')
plt.show()

# ===========================
# 5. Correlation Heatmap
# ===========================
numeric_df = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()

print("EDA completed successfully!")