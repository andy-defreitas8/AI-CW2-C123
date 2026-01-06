import os
import glob
import pandas as pd

# Extract label from image name
def extract_label(image_name):
    # Example: "J_sample_9" → "J"
    return image_name.split("_")[0]

# Load dataset and combine all CSVs into a single DataFrame
def load_landmark_folder(folder_path):
    dfs = []

    for path in glob.glob(os.path.join(folder_path, "*.csv")):
        df = pd.read_csv(path)

        image_name = df.iloc[0, 0]
        label = extract_label(image_name)

        df["label"] = label
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

df = load_landmark_folder("C:/Users/Andy/Downloads/CW2_landmarks")
save_path = "C:/Users/Andy/Downloads/combined_landmarks.csv"
df.to_csv(save_path, index=False)