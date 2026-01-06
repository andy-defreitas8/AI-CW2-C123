import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
IMAGE_DIR = "C:/Users/Andy/Downloads/CW2_dataset_final/J"
OUTPUT_DIR = "C:/Users/Andy/Downloads/CW2_landmarks"
MAX_HANDS = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Initialize MediaPipe Hands
# -------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=0.5
)

# -------------------------------
# Generate column names
# -------------------------------
def generate_hand_columns():
    columns = ["image_name"]
    for hand_idx in range(MAX_HANDS):
        for lm_idx in range(21):
            columns.extend([
                f"hand{hand_idx}_lm{lm_idx}_x",
                f"hand{hand_idx}_lm{lm_idx}_y",
                f"hand{hand_idx}_lm{lm_idx}_z",
            ])
    return columns

columns = generate_hand_columns()

# -------------------------------
# Process images
# -------------------------------
for filename in sorted(os.listdir(IMAGE_DIR)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read {filename}")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Initialize row with NaNs
    landmark_data = np.full((MAX_HANDS, 21, 3), np.nan)

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= MAX_HANDS:
                break
            for lm_idx, lm in enumerate(hand_landmarks.landmark):
                landmark_data[hand_idx, lm_idx] = [lm.x, lm.y, lm.z]

    # Build row
    row = [filename] + landmark_data.flatten().tolist()

    # Create DataFrame (single-row)
    df = pd.DataFrame([row], columns=columns)

    # Save CSV with same base name as image
    csv_name = os.path.splitext(filename)[0] + ".csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    df.to_csv(csv_path, index=False)

    print(f"Saved {csv_path}")
