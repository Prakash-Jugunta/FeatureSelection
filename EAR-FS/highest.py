import re

def extract_highest_acc_rate(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    acc_rates = []

    for line in lines:
        match = re.search(r'acc rate:\s*([0-9.]+)', line)
        if match:
            acc_rates.append(float(match.group(1)))

    if acc_rates:
        highest_acc = max(acc_rates)
        print(f"Highest acc rate: {highest_acc}")
    else:
        print("No accuracy rates found in the file.")

# Example usage
extract_highest_acc_rate('log_hybrid.txt')  # Replace with your file name
