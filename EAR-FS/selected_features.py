import torch
import numpy as np

# --- Step 1: Load and flatten the selection vector
select_rate = torch.load('select_rate.pt').detach().cpu().numpy().flatten()

# --- Step 2: Sort features by importance
sorted_indices = np.argsort(-select_rate)

# --- Step 3: Load PSO’s best feature count (update folder name as needed)
record = np.load('./42/record.npy')  # change '42' to your seed

best_feature_count = int(record[1])

# --- Step 4: Select top features
selected_features = sorted_indices[:best_feature_count]

# --- Step 5: Create 0/1 vector
binary_selection_vector = np.zeros_like(select_rate)
binary_selection_vector[selected_features] = 1

# --- Step 6: Print info
print("🔢 Total features before selection:", len(select_rate))
print("✅ Features selected by PSO:", len(selected_features))
print("📍Selected feature indices:", selected_features.tolist())
print("🧠 Binary selection vector (first 50):", binary_selection_vector[:50])

# --- Optional: Save
np.save("final_selected_feature_indices.npy", selected_features)
np.save("final_selection_mask.npy", binary_selection_vector)
