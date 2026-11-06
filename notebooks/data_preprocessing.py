import pandas as pd

# Load the dataset
data = pd.read_csv("dataset/ai4i2020.csv")

# Select useful input features
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Select input data (X)
X = data[features]

# Select target variable (y)
y = data["Machine failure"]

# Display the selected data
print("Input Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

print("\nInput Data Shape:", X.shape)
print("Target Data Shape:", y.shape)
# Convert categorical Type column into numerical values
X = pd.get_dummies(X, columns=["Type"], dtype=int)

print("\nEncoded Input Features:")
print(X.head())

print("\nUpdated Input Data Shape:", X.shape)
# Split dataset into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining and Testing Data:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nFailure distribution in training data:")
print(y_train.value_counts())

print("\nFailure distribution in testing data:")
print(y_test.value_counts())