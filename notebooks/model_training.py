import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# Load dataset
data = pd.read_csv("dataset/ai4i2020.csv")

# Select input features
features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = data[features]
y = data["Machine failure"]

# Convert product type into numerical values
X = pd.get_dummies(X, columns=["Type"], dtype=int)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

# Train model
model.fit(X_train, y_train)

# Predict machine failures
y_pred = model.predict(X_test)

# Evaluate model
print("\nRandom Forest Model Results")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save trained model
joblib.dump(model, "models/random_forest_model.pkl")

print("\nModel saved successfully!")
# Logistic Regression Model

logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
)

# Train model
logistic_model.fit(X_train, y_train)

# Make predictions
logistic_pred = logistic_model.predict(X_test)

# Evaluate model
print("\nLogistic Regression Model Results")

print("\nAccuracy:")
print(accuracy_score(y_test, logistic_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_pred))

# Save model
joblib.dump(
    logistic_model,
    "models/logistic_regression_model.pkl"
)

print("\nLogistic Regression Model Saved Successfully!")
# XGBoost Model

# Calculate class imbalance ratio
scale_pos_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

# Create XGBoost model
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

# Clean feature names for XGBoost
X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()

X_train_xgb.columns = X_train_xgb.columns.str.replace(
    r"[\[\]<>]", "_", regex=True
)

X_test_xgb.columns = X_test_xgb.columns.str.replace(
    r"[\[\]<>]", "_", regex=True
)

# Train model
xgb_model.fit(X_train_xgb, y_train)

# Make predictions
xgb_pred = xgb_model.predict(X_test_xgb)
# Evaluate model
print("\nXGBoost Model Results")

print("\nAccuracy:")
print(accuracy_score(y_test, xgb_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        xgb_pred,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))

# Save model
joblib.dump(
    xgb_model,
    "models/xgboost_model.pkl"
)

print("\nXGBoost Model Saved Successfully!")
# Model Comparison

models = {
    "Random Forest": (model, y_pred),
    "Logistic Regression": (logistic_model, logistic_pred),
    "XGBoost": (xgb_model, xgb_pred)
}

comparison_results = []

for name, (trained_model, predictions) in models.items():

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    accuracy = accuracy_score(y_test, predictions)

    # Calculate PR-AUC using prediction probabilities
    if name == "XGBoost":
        probabilities = trained_model.predict_proba(X_test_xgb)[:, 1]
    else:
        probabilities = trained_model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(
        y_test, probabilities
    )

    comparison_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "PR-AUC": pr_auc
    })

# Create comparison table
comparison_df = pd.DataFrame(comparison_results)

print("\nFINAL MODEL COMPARISON")
print(comparison_df.round(4))

# Save comparison report
comparison_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nModel comparison report saved successfully!")