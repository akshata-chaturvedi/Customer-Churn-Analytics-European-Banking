import pandas as pd

# Load dataset
df = pd.read_csv("European_Bank.csv")

# Display dataset information
print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Fill missing numerical values with median
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with mode
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove unnecessary columns
df.drop(['Year', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Save cleaned dataset
df.to_csv("cleaned_data.csv", index=False)

print("\nData Cleaning Completed Successfully!")
print("Cleaned file saved as cleaned_data.csv")