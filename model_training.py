import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("cleaned_data.csv")

# Convert categorical columns into numbers
le = LabelEncoder()

df['Geography'] = le.fit_transform(df['Geography'])
df['Gender'] = le.fit_transform(df['Gender'])

# Features and Target
X = df.drop('Exited', axis=1)
y = df['Exited']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Logistic Regression
# ======================
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:")
print(round(lr_accuracy * 100, 2), "%")

# ======================
# Random Forest
# ======================
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:")
print(round(rf_accuracy * 100, 2), "%")
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
joblib.dump(lr_model, "logistic_model.pkl")
print("Logistic Regression model saved successfully!")
joblib.dump(rf_model, "random_forest_model.pkl")
