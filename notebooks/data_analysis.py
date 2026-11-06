import pandas as pd

# Load the industrial dataset
data = pd.read_csv("dataset/ai4i2020.csv")

# Display first five records
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Display number of rows and columns
print("\nDataset Shape:")
print(data.shape)
# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# Check statistical information
print("\nStatistical Summary:")
print(data.describe())
import matplotlib.pyplot as plt
import seaborn as sns

# Count machine failures
failure_counts = data["Machine failure"].value_counts()

print("\nMachine Failure Distribution:")
print(failure_counts)

# Create a bar chart
plt.figure(figsize=(6, 4))

sns.countplot(
    x="Machine failure",
    data=data
)

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Number of Machines")

plt.show()
# Calculate machine failure percentage

total_machines = len(data)
failed_machines = data["Machine failure"].sum()
failure_percentage = (failed_machines / total_machines) * 100

print("\nTotal Machines:", total_machines)
print("Failed Machines:", failed_machines)
print("Failure Percentage:", round(failure_percentage, 2), "%")
# Machine Failure by Product Type

print("\nMachine Failure by Product Type:")

failure_by_type = data.groupby("Type")["Machine failure"].sum()

print(failure_by_type)

# Create bar chart

plt.figure(figsize=(7, 4))

sns.barplot(
    x=failure_by_type.index,
    y=failure_by_type.values
)

plt.title("Machine Failures by Product Type")
plt.xlabel("Product Type")
plt.ylabel("Number of Failures")

plt.show()
# Failure percentage by product type

failure_percentage_by_type = (
    data.groupby("Type")["Machine failure"].mean() * 100
)

print("\nFailure Percentage by Product Type:")
print(failure_percentage_by_type.round(2))
# Tool Wear Analysis
plt.figure(figsize=(7, 4))

sns.boxplot(
    x="Machine failure",
    y="Tool wear [min]",
    data=data
)

plt.title("Tool Wear vs Machine Failure")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Tool Wear (minutes)")

plt.show()
# Rotational Speed Analysis
plt.figure(figsize=(7, 4))

sns.boxplot(
    x="Machine failure",
    y="Rotational speed [rpm]",
    data=data
)

plt.title("Rotational Speed vs Machine Failure")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Rotational Speed (RPM)")

plt.show()
# Torque Analysis
plt.figure(figsize=(7, 4))

sns.boxplot(
    x="Machine failure",
    y="Torque [Nm]",
    data=data
)

plt.title("Torque vs Machine Failure")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Torque (Nm)")

plt.show()
# Air Temperature Analysis
plt.figure(figsize=(7, 4))

sns.boxplot(
    x="Machine failure",
    y="Air temperature [K]",
    data=data
)

plt.title("Air Temperature vs Machine Failure")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Air Temperature (Kelvin)")

plt.show()
# Process Temperature Analysis
plt.figure(figsize=(7, 4))

sns.boxplot(
    x="Machine failure",
    y="Process temperature [K]",
    data=data
)

plt.title("Process Temperature vs Machine Failure")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Process Temperature (Kelvin)")

plt.show()
# Correlation Heatmap

numeric_data = data.select_dtypes(include=["number"])

# Remove ID columns and failure-mode columns
numeric_data = numeric_data.drop(
    columns=["UDI", "TWF", "HDF", "PWF", "OSF", "RNF"]
)

correlation = numeric_data.corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Machine Sensor Data")
plt.tight_layout()
plt.show()