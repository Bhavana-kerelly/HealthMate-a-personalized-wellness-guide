import pandas as pd
import random

# Define the number of records
num_records = 7000

# Generate synthetic data
data = {
    "Age": [random.randint(18, 65) for _ in range(num_records)],
    "Gender": [random.choice(["Male", "Female"]) for _ in range(num_records)],
    "BMI": [round(random.uniform(18.5, 35.0), 1) for _ in range(num_records)],
    "Daily Water Intake (L)": [round(random.uniform(0.5, 4.0), 1) for _ in range(num_records)],
    "Physical Activity Level": [random.choice(["Low", "Moderate", "High"]) for _ in range(num_records)],
    "Eating Habits": [random.choice(["Healthy", "Moderate", "Unhealthy"]) for _ in range(num_records)],
    "Medical Conditions": [random.choice(["None", "Diabetes", "Hypertension", "Anemia"]) for _ in range(num_records)],
    "Medication Usage": [random.choice(["None", "Occasionally", "Regularly"]) for _ in range(num_records)],
    "Sleep Hours": [random.randint(4, 10) for _ in range(num_records)],
    "Stress Level": [random.choice(["Low", "Moderate", "High"]) for _ in range(num_records)],
}

# Define deficiency types and corresponding food recommendations
deficiency_food_map = {
    "None": "Maintain a balanced diet",
    "Iron Deficiency": "Spinach, Red Meat, Lentils, Tofu",
    "Vitamin D Deficiency": "Salmon, Dairy Products, Mushrooms, Sunlight",
    "Magnesium Deficiency": "Nuts, Whole Grains, Dark Chocolate, Avocados",
    "Calcium Deficiency": "Milk, Cheese, Yogurt, Leafy Greens",
}

# Generate deficiency output
deficiency_types = list(deficiency_food_map.keys())
data["Deficiency"] = [random.choice(deficiency_types) for _ in range(num_records)]

# Convert deficiency into binary Yes/No
data["Deficiency_Detected"] = ["Yes" if d != "None" else "No" for d in data["Deficiency"]]

# Assign food recommendations based on deficiency type
data["Recommended_Food"] = [deficiency_food_map[d] for d in data["Deficiency"]]

# Create DataFrame
df = pd.DataFrame(data)

# Save the dataset locally
df.to_csv("nutritional_dataset.csv", index=False)

print("Dataset saved as 'nutritional_dataset.csv' with food recommendations.")
