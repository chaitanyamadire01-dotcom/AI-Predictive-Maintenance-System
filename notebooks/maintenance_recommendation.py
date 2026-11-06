import pandas as pd
import joblib

print("Libraries imported successfully!")

# Load the trained XGBoost model
xgb_model = joblib.load("models/xgboost_model.pkl")

print("\nXGBoost model loaded successfully!")
print("Model:", type(xgb_model).__name__)
# Sample machine sensor readings
machine_data = {
    "Air temperature [K]": 303.5,
    "Process temperature [K]": 313.5,
    "Rotational speed [rpm]": 1400,
    "Torque [Nm]": 50.0,
    "Tool wear [min]": 200,
    "Type_H": 0,
    "Type_L": 1,
    "Type_M": 0
}

# Convert sensor readings into a DataFrame
input_data = pd.DataFrame([machine_data])

print("\nMachine sensor data prepared successfully!")
print(input_data)
# Clean feature names for XGBoost
input_data.columns = input_data.columns.str.replace(
    r"[\[\]<>]", "_", regex=True
)

# Predict machine failure
prediction = xgb_model.predict(input_data)

# Calculate failure probability
probability = xgb_model.predict_proba(input_data)[0][1]

print("\nMACHINE FAILURE PREDICTION")

if prediction[0] == 1:
    print("Prediction: Machine Failure Detected")
else:
    print("Prediction: No Machine Failure Detected")

print("Failure Probability:", round(probability * 100, 2), "%")
# Maintenance recommendation system

recommendations = []

# Check tool wear
if machine_data["Tool wear [min]"] > 180:
    recommendations.append(
        "High tool wear detected: Inspect or replace the cutting tool."
    )

# Check torque
if machine_data["Torque [Nm]"] > 50:
    recommendations.append(
        "High torque detected: Inspect motor load and mechanical components."
    )

# Check rotational speed
if machine_data["Rotational speed [rpm]"] < 1500:
    recommendations.append(
        "Low rotational speed detected: Inspect motor and speed controller."
    )

# Check air temperature
if machine_data["Air temperature [K]"] > 305:
    recommendations.append(
        "High air temperature detected: Inspect the cooling system."
    )

# Check process temperature
if machine_data["Process temperature [K]"] > 315:
    recommendations.append(
        "High process temperature detected: Inspect temperature control."
    )

# Display recommendations
print("\nMAINTENANCE RECOMMENDATIONS")

if recommendations:
    for item in recommendations:
        print("-", item)
else:
    print("No immediate maintenance action suggested by these prototype rules.")
    # Assign maintenance priority

print("\nMAINTENANCE PRIORITY")

if prediction[0] == 1 or probability >= 0.70:
    priority = "HIGH"
    print("Priority:", priority)
    print("Action: Immediate machine inspection required.")

elif probability >= 0.30 or len(recommendations) >= 2:
    priority = "MEDIUM"
    print("Priority:", priority)
    print("Action: Schedule maintenance inspection.")

else:
    priority = "LOW"
    print("Priority:", priority)
    print("Action: Continue regular machine monitoring.")
    # Save maintenance report

report = pd.DataFrame([{
    "Failure Prediction": (
        "Failure Detected" if prediction[0] == 1
        else "No Failure Detected"
    ),
    "Failure Probability (%)": round(probability * 100, 2),
    "Maintenance Priority": priority,
    "Recommendations": " | ".join(recommendations)
        if recommendations else "No recommendations"
}])

report.to_csv(
    "reports/maintenance_report.csv",
    index=False
)

print("\nMaintenance report saved successfully!")
print("File: reports/maintenance_report.csv")