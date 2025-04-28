import os
import numpy as np
import torch

# --- CONFIG ---
PARENT_DIR = '.'  # where your seed subfolders (named '0', '1', etc.) are
CLASSIFIERS = ['knn', 'svm', 'decision_tree', 'random_forest']
ROOT_SELECT_RATE = 'select_rate.pt'  # path to select_rate.pt

def find_seed_dirs(parent):
    return sorted(
        [d for d in os.listdir(parent)
         if os.path.isdir(os.path.join(parent, d)) and d.isdigit()],
        key=int
    )

def load_record(seed_dir):
    rec_path = os.path.join(seed_dir, 'record.npy')
    if not os.path.exists(rec_path):
        raise FileNotFoundError(f"Missing {rec_path}")
    return np.load(rec_path)

def aggregate(parent_dir):
    seed_dirs = find_seed_dirs(parent_dir)
    max_info = {clf: {'acc': -np.inf, 'k': None} for clf in CLASSIFIERS}  # Track max acc and corresponding k

    for sd in seed_dirs:
        rec = load_record(os.path.join(parent_dir, sd))
        for i, clf in enumerate(CLASSIFIERS):
            acc = rec[2*i]
            k = int(rec[2*i + 1])
            if acc > max_info[clf]['acc']:  # Update if better accuracy
                max_info[clf]['acc'] = acc
                max_info[clf]['k'] = k

    return max_info

def main():
    # 1) Aggregate best accuracies and their corresponding k
    max_info = aggregate(PARENT_DIR)

    # 2) Print all classifiers' best accuracies and corresponding k
    print("=== Highest Accuracies for Each Classifier ===")
    for clf in CLASSIFIERS:
        acc = max_info[clf]['acc']
        k = max_info[clf]['k']
        print(f"{clf:<15} | Highest Accuracy: {acc:.4f} | Corresponding k: {k}")

    # 3) Find overall best classifier
    best_clf = max(max_info, key=lambda c: max_info[c]['acc'])
    best_acc = max_info[best_clf]['acc']
    best_k = max_info[best_clf]['k']

    # 4) Load select_rate.pt and extract top-k
    if not os.path.exists(ROOT_SELECT_RATE):
        raise FileNotFoundError(f"Cannot find {ROOT_SELECT_RATE}")
    sr_tensor = torch.load(ROOT_SELECT_RATE, weights_only=True)
    sr = sr_tensor.detach().cpu().numpy()
    top_indices = np.argsort(-sr)[:best_k].tolist()

    # 5) Print best classifier summary
    print("\n=== Overall Best Classifier ===")
    print(f"Best Classifier    : {best_clf}")
    print(f"Highest Accuracy   : {best_acc:.4f}")
    print(f"Corresponding k    : {best_k}")

if __name__ == '__main__':
    main()
