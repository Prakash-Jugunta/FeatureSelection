import torch
import numpy as np

# Load feature selection weights
select_rate = torch.load("select_rate.pt", weights_only=True)

# Convert to NumPy for easy handling
feature_scores = select_rate.flatten().detach().cpu().numpy()

# Print the top 10 most important features
top_features = np.argsort(feature_scores)[::-1]  # Sort in descending order
print("🔹 Top 10 Most Important Features (Index Positions):", top_features[:10])

# Print their importance scores
print("🔹 Corresponding Importance Scores:", feature_scores[top_features[:10]])
