import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
# Seaborn has the Titanic dataset built-in
df = sns.load_dataset('titanic')

print("First 5 rows of the data:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Data Cleaning

# Drop columns that are not useful or have too many missing values
df = df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'], axis=1)

# Fill missing age with median
df['age'].fillna(df['age'].median(), inplace=True)

# Fill missing embark with most common value
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Convert categorical columns to numbers
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("\nAfter cleaning - Missing values:")
print(df.isnull().sum())
print("\nCleaned data sample:")
print(df.head())

# Exploratory Data Analysis (EDA)

# Survival count
plt.figure(figsize=(6,4))
sns.countplot(x='survived', data=df)
plt.title('Survival Count (0 = Died, 1 = Survived)')
plt.show()

# Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='sex', hue='survived', data=df)
plt.title('Survival by Gender')
plt.show()

# Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x='pclass', hue='survived', data=df)
plt.title('Survival by Passenger Class')
plt.show()

# Age distribution
plt.figure(figsize=(8,4))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Prepare data for model

X = df.drop('survived', axis=1)
y = df['survived']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Build the model (Logistic Regression)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions and evaluate

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Simple conclusion
print("\n--- Project Conclusion ---")
print("The model predicts survival with reasonable accuracy.")
print("Women and higher class passengers had better survival chances.")
print("This is a basic classification project using Logistic Regression.")