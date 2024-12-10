import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define your file paths
file_path_1 = r"File paht "  # Gender submission file
file_path_2 = r"File paht"  # Test data
file_path_3 = r"File paht"  # Training data

# Load the data from the files
train_data = pd.read_csv(file_path_3)  # Training data
test_data = pd.read_csv(file_path_2)  # Test data

# Preview the training dataset
print("Training Data Preview:")
print(train_data.head())

# Preprocessing the training data
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]  # Adjust if needed
target = "Survived"

# Handle missing values in training data
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())  # Fill missing ages with median
train_data["Embarked"] = train_data["Embarked"].fillna(train_data["Embarked"].mode()[0])  # Fill missing Embarked with mode

# Convert categorical columns to numerical using one-hot encoding
train_data = pd.get_dummies(train_data, columns=["Sex", "Embarked"], drop_first=True)

# Preview the updated DataFrame after encoding
print("Data After One-Hot Encoding:")
print(train_data.head())

# Prepare features (X) and target (y) for training
X = train_data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]]
y = train_data[target]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Preprocessing the test data
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median())
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())  # Handle missing Fare in test set

# Convert categorical columns to numerical using one-hot encoding (same as for training data)
test_data = pd.get_dummies(test_data, columns=["Sex", "Embarked"], drop_first=True)

# Ensure the test data has the same columns as the training data
X_test = test_data[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_male", "Embarked_Q", "Embarked_S"]]

# Align columns to match between train and test data (handling any missing columns)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Make predictions on the test set
test_data["Survived"] = model.predict(X_test)

# Save predictions to a CSV file
output_file_path = r"File paht"
test_data[["PassengerId", "Survived"]].to_csv(output_file_path, index=False)
print(f"Predictions saved to {output_file_path}")
