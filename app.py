import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="AI Predictive Maintenance",
    page_icon="🏭",
    layout="wide"
)

# Dashboard title
st.title("🏭 AI-Driven Predictive Maintenance System")

st.subheader(
    "Machine Failure Prediction and Maintenance Recommendation"
)

st.write(
    "Monitor machine sensor readings and predict potential failures "
    "using Machine Learning."
)

st.divider()

# Load trained XGBoost model
xgb_model = joblib.load("models/xgboost_model.pkl")

st.success("Machine Learning Model Loaded Successfully!")
# Machine sensor input section

st.header("⚙️ Machine Sensor Inputs")

col1, col2 = st.columns(2)

with col1:
    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=303.5
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=313.5
    )

    rotational_speed = st.number_input(
        "Rotational Speed (RPM)",
        min_value=0,
        max_value=5000,
        value=1400
    )

with col2:
    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    tool_wear = st.number_input(
        "Tool Wear (Minutes)",
        min_value=0,
        max_value=300,
        value=200
    )

    product_type = st.selectbox(
        "Product Type",
        ["H", "L", "M"]
    )

st.write("Enter the machine sensor readings above.")
# Display sensor readings

st.divider()
st.subheader("📊 Current Machine Sensor Readings")

sensor_col1, sensor_col2, sensor_col3 = st.columns(3)

sensor_col1.metric("Air Temperature", f"{air_temp} K")
sensor_col2.metric("Process Temperature", f"{process_temp} K")
sensor_col3.metric("Rotational Speed", f"{rotational_speed} RPM")

sensor_col4, sensor_col5 = st.columns(2)

sensor_col4.metric("Torque", f"{torque} Nm")
sensor_col5.metric("Tool Wear", f"{tool_wear} minutes")
# Sensor Visualization

st.divider()
st.subheader("📈 Machine Sensor Visualization")

sensor_chart_data = pd.DataFrame({
    "Sensor": [
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear"
    ],
    "Reading": [
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear
    ]
})

st.bar_chart(
    sensor_chart_data,
    x="Sensor",
    y="Reading"
)
# Predict machine failure

st.divider()

if st.button("🔍 Predict Machine Failure", type="primary"):

    # Prepare input data
    machine_data = {
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_H": 1 if product_type == "H" else 0,
        "Type_L": 1 if product_type == "L" else 0,
        "Type_M": 1 if product_type == "M" else 0
    }

    input_data = pd.DataFrame([machine_data])

    # Clean feature names for XGBoost
    input_data.columns = input_data.columns.str.replace(
        r"[\[\]<>]", "_", regex=True
    )

    # Predict failure
    prediction = xgb_model.predict(input_data)
    probability = xgb_model.predict_proba(input_data)[0][1]

    st.subheader("Machine Failure Prediction")

    st.metric(
        "Failure Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction[0] == 1:
        st.error("⚠️ Machine Failure Detected!")
    else:
        st.success("✅ No Machine Failure Detected")
            # Maintenance recommendations

    recommendations = []

    if tool_wear > 180:
        recommendations.append(
            "High tool wear: Inspect or replace the cutting tool."
        )

    if torque > 50:
        recommendations.append(
            "High torque: Inspect motor load and mechanical components."
        )

    if rotational_speed < 1500:
        recommendations.append(
            "Low rotational speed: Inspect motor and speed controller."
        )

    if air_temp > 305:
        recommendations.append(
            "High air temperature: Inspect the cooling system."
        )

    if process_temp > 315:
        recommendations.append(
            "High process temperature: Inspect temperature control."
        )

    st.subheader("Maintenance Recommendations")

    if recommendations:
        for item in recommendations:
            st.warning(item)
    else:
        st.info("No maintenance recommendations from the current prototype rules.")
            # Maintenance priority

    st.subheader("Maintenance Priority")

    if prediction[0] == 1 or probability >= 0.70:
        priority = "HIGH PRIORITY"
        st.error(priority)
        st.write("Immediate machine inspection required.")

    elif probability >= 0.30 or len(recommendations) >= 2:
        priority = "MEDIUM PRIORITY"
        st.warning(priority)
        st.write("Schedule maintenance inspection.")

    else:
        priority = "LOW PRIORITY"
        st.success(priority)
        st.write("Continue regular machine monitoring.")

    # Download Maintenance Report

    report_data = {
        "Air Temperature (K)": air_temp,
        "Process Temperature (K)": process_temp,
        "Rotational Speed (RPM)": rotational_speed,
        "Torque (Nm)": torque,
        "Tool Wear (Minutes)": tool_wear,
        "Product Type": product_type,
        "Failure Probability (%)": round(probability * 100, 2),
        "Prediction": "Failure Detected" if prediction[0] == 1 else "No Failure",
        "Maintenance Recommendations": "; ".join(recommendations),
        "Maintenance Priority": priority
    }

    report_df = pd.DataFrame([report_data])

    csv = report_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Maintenance Report",
        data=csv,
        file_name="maintenance_report.csv",
        mime="text/csv"
    )
    # Model Performance

st.divider()
st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", "95.70%")
col2.metric("Precision", "43.38%")
col3.metric("Recall", "86.76%")
col4.metric("F1 Score", "57.84%")

st.info(
    "Performance values are from the XGBoost model evaluated "
    "on the held-out test dataset."
)
# Confusion Matrix

st.divider()
st.header("🎯 Confusion Matrix")

confusion_matrix_data = pd.DataFrame(
    [[1855, 77], [9, 59]],
    index=["Actual No Failure", "Actual Failure"],
    columns=["Predicted No Failure", "Predicted Failure"]
)

st.write(confusion_matrix_data)

st.caption(
    "Confusion matrix for the XGBoost model on the test dataset."
)
# Project Information

st.divider()
st.header("ℹ️ About This Project")

st.write("""
**Project Title:** AI-Driven Predictive Maintenance System
Using Machine Learning and Industrial Data Analytics

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

**Machine Learning Algorithm:** XGBoost

**Explainable AI:** SHAP

**Dashboard:** Streamlit

**Programming Language:** Python

**Objective:** To predict potential machine failures using
machine sensor readings and provide maintenance recommendations.
""")
# Explainable AI - SHAP

st.divider()
st.header("🧠 Explainable AI (SHAP)")

st.subheader("Feature Importance")
st.image(
    "reports/shap_feature_importance.png",
    caption="SHAP Feature Importance"
)

st.subheader("SHAP Summary Plot")
st.image(
    "reports/shap_summary.png",
    caption="SHAP Summary Plot"
)
# Model Details

st.divider()
st.header("🤖 Model Information")

st.write("Selected Model: XGBoost Classifier")

st.write("Dataset Size: 10,000 machine records")

st.write("Training Records: 8,000")

st.write("Testing Records: 2,000")

st.write("Number of Input Features: 8")

st.write("Target Variable: Machine Failure")

st.write("Class Imbalance: 3.39% machine failures in the dataset")
# Machine Health Status

st.divider()
st.header("🟢 Machine Health Status")

if tool_wear >= 250 or torque >= 65 or rotational_speed <= 1200 or air_temp >= 310 or process_temp >= 320:
    st.error("Machine Health: Critical")
    st.write("Immediate machine inspection is required.")

elif tool_wear > 180 or torque > 50 or rotational_speed < 1500:
    st.warning("Machine Health: Needs Attention")
    st.write("Schedule a maintenance inspection.")

else:
    st.success("Machine Health: Normal")
    st.write("Continue regular machine monitoring.")