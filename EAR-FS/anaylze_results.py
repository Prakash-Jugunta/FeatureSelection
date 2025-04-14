import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Automatically detect all random seed folders
seed_folders = [folder for folder in os.listdir() if folder.isdigit()]
seed_folders = sorted(seed_folders, key=int)  # Sort numerically

# Classifiers to analyze
classifiers = ["knn", "svm", "decision_tree", "random_forest"]

# Store the best results
best_results = {}

print(f"📂 Found {len(seed_folders)} random seed folders: {seed_folders}\n")

for random_seed in seed_folders:
    print(f"\n🔍 Analyzing results for Random Seed: {random_seed}")

    for name in classifiers:
        # ✅ 1️⃣ Load & Plot Feature Selection Performance
        prop_path = f'./{random_seed}/prop/{name}y.npy'
        if os.path.exists(prop_path):
            y = np.load(prop_path)
            x = np.arange(0.02, 1.01, 0.02)

            plt.figure(figsize=(8, 5))
            plt.plot(x, y, marker='o', linestyle='-')
            plt.xlabel("Feature Proportion")
            plt.ylabel("Accuracy")
            plt.title(f"Feature Selection Performance for {name} (Seed {random_seed})")
            plt.grid()
        else:
            print(f"❌ {prop_path} not found! Skipping {name} for Seed {random_seed}.")

        # ✅ 2️⃣ Compare Classifier Performance
        accuracy_path = f'./{random_seed}/y/{name}y.npy'
        if os.path.exists(accuracy_path):
            y = np.load(accuracy_path)
            best_acc = max(y)
            best_results[(random_seed, name)] = best_acc  # Store best accuracy

            print(f"✅ Best Accuracy for {name} (Seed {random_seed}): {best_acc:.4f}")
        else:
            print(f"❌ {accuracy_path} not found! Skipping {name} for Seed {random_seed}.")

        # ✅ 3️⃣ Display Saved Plots
        png_path = f'./{random_seed}/png/{name}_subset_prop.png'
        if os.path.exists(png_path):
            img = mpimg.imread(png_path)
            plt.figure(figsize=(6, 4))
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"Feature Selection Plot: {name} (Seed {random_seed})")
        else:
            print(f"❌ {png_path} not found! Skipping {name} for Seed {random_seed}.")

# ✅ 4️⃣ Print Overall Best Results
print("\n📊 Summary of Best Accuracy Across Seeds:")
for (seed, model), acc in sorted(best_results.items(), key=lambda x: x[1], reverse=True):
    print(f"🌟 Seed {seed} | Model: {model} | Accuracy: {acc:.4f}")

# Save results to file
np.save("best_results.npy", best_results)
print("\n✅ Saved best results to best_results.npy")
