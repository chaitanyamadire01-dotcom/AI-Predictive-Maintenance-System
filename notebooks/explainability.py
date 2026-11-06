import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

print("Libraries imported successfully!")

# Load dataset
data = pd.read_csv("dataset/ai4i2020.csv")

print("\nDataset loaded successfully!")
print("Dataset Shape:", data.shape)
# Select useful input features
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

# Separate input features and target
X = data[features]
y = data["Machine failure"]

# Convert categorical Type into numerical values
X = pd.get_dummies(X, columns=["Type"], dtype=int)

print("\nFeatures selected successfully!")
print("Input Data Shape:", X.shape)
print("Target Data Shape:", y.shape)

print("\nFeature Names:")
print(X.columns.tolist())
# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset split successfully!")

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Clean feature names for XGBoost
X_train.columns = X_train.columns.str.replace(
    r"[\[\]<>]", "_", regex=True
)

X_test.columns = X_test.columns.str.replace(
    r"[\[\]<>]", "_", regex=True
)

print("\nFeature names prepared for XGBoost!")
print(X_train.columns.tolist())
# Load the trained XGBoost model
xgb_model = joblib.load("models/xgboost_model.pkl")

print("\nXGBoost model loaded successfully!")

# Check model type
print("Model:", type(xgb_model).__name__)
# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values for testing data
shap_values = explainer(X_test)

print("\nSHAP values calculated successfully!")

print("SHAP explanation shape:", shap_values.values.shape)
# Generate SHAP feature importance plot
plt.figure()

shap.plots.bar(shap_values, show=False)

plt.title("SHAP Feature Importance")

plt.tight_layout()

# Save the graph
plt.savefig(
    "reports/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nSHAP feature importance graph saved successfully!")

plt.show()
# Generate SHAP summary plot
plt.figure()

shap.plots.beeswarm(shap_values, show=False)

plt.title("SHAP Summary Plot")

plt.tight_layout()

# Save the graph
plt.savefig(
    "reports/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nSHAP summary graph saved successfully!")

plt.show()