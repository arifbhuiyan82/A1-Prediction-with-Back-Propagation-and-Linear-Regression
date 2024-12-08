import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
file_path = r'C:\Users\Arif Bhuiyan\Desktop\A1 Final\filtered_cybersecurity_attacks.csv'
df = pd.read_csv(file_path)

# Check and display missing values in each column
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        # For categorical columns, fill missing with "Unknown"
        df[col].fillna("Unknown", inplace=True)
    else:
        # For numeric columns, fill missing with the column's mean
        df[col].fillna(df[col].mean(), inplace=True)

# Define numeric and categorical columns
numerical_columns = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores"]
categorical_columns = [
    "Protocol", "Packet Type", "Traffic Type", 
    "Malware Indicators", "Attack Type", 
    "Action Taken", "Severity Level", "Log Source"
]

# Encode categorical columns using LabelEncoder
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for future inverse transformation if needed

# Normalize the numeric columns
scaler = StandardScaler()  
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Check for outliers in numeric columns
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    print(f"Outliers detected in {col}:\n", outliers)

# Save the processed dataset
processed_file_path = r'C:\Users\Arif Bhuiyan\Desktop\A1 Final\processed_cybersecurity_attacks.csv'
df.to_csv(processed_file_path, index=False)
print(f"Processed dataset saved to {processed_file_path}")
