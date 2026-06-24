import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")

# ======================
# Age Group Segmentation
# ======================
bins = [18,30,40,50,60,100]
labels = ['18-30','31-40','41-50','51-60','60+']

df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

plt.figure(figsize=(8,5))
sns.countplot(x='AgeGroup', data=df)
plt.title('Customer Age Group Segmentation')
plt.savefig('age_group_segmentation.png')
plt.show()

# ==========================
# Credit Score Segmentation
# ==========================
credit_bins = [300,500,650,750,900]
credit_labels = ['Poor','Average','Good','Excellent']

df['CreditCategory'] = pd.cut(df['CreditScore'],
                              bins=credit_bins,
                              labels=credit_labels)

plt.figure(figsize=(8,5))
sns.countplot(x='CreditCategory', data=df)
plt.title('Credit Score Segmentation')
plt.savefig('credit_score_segmentation.png')
plt.show()

# ==========================
# Balance Segmentation
# ==========================
plt.figure(figsize=(8,5))
sns.histplot(df['Balance'], bins=20)
plt.title('Balance Distribution')
plt.savefig('balance_distribution.png')
plt.show()

# ==========================
# Active Members
# ==========================
plt.figure(figsize=(6,4))
sns.countplot(x='IsActiveMember', data=df)
plt.title('Active vs Inactive Members')
plt.savefig('active_members.png')
plt.show()

# ==========================
# Product Usage
# ==========================
plt.figure(figsize=(6,4))
sns.countplot(x='NumOfProducts', data=df)
plt.title('Product Usage Distribution')
plt.savefig('product_usage.png')
plt.show()

print("Customer Segmentation Analysis Completed!")