import pandas as pd

# Path to the input CSV file
input_file = "C:\\Users\\Andy\\Downloads\\combined_landmarks.csv"

# Path to the output CSV file (cleaned version)
output_file = "C:\\Users\\Andy\\Downloads\\combined_landmarks_clean.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file)

# Identify the landmark columns (hand0_lm0_x to hand0_lm20_z)
landmark_cols = [f'hand0_lm{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]

# Remove rows where any landmark column has a blank (NaN) value
df_clean = df.dropna(subset=landmark_cols)

# Write the cleaned DataFrame to a new CSV file
df_clean.to_csv(output_file, index=False)

print(f"Cleaned CSV saved to {output_file}")