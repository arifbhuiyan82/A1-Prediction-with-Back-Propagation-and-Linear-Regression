import pandas as pd

# File paths
input_file = r'C:\Users\Arif Bhuiyan\Desktop\A1 Final\cybersecurity_attacks.csv'
output_file = r'C:\Users\Arif Bhuiyan\Desktop\A1 Final\filtered_cybersecurity_attacks.csv'

# Load the dataset
df = pd.read_csv(input_file)

# Restrict to a maximum of 5,000 rows
df = df.head(5000)

# Select only the required columns
required_columns = [
    'Source Port', 'Destination Port', 'Protocol', 'Packet Length', 
    'Packet Type', 'Traffic Type', 'Malware Indicators', 
    'Attack Type', 'Action Taken', 'Severity Level', 'Log Source', 
    'Anomaly Scores'
]
df_filtered = df[required_columns]

# Save the filtered dataframe to a new CSV file
df_filtered.to_csv(output_file, index=False)

print(f"Filtered dataset with a maximum of 15,000 rows saved to {output_file}")
