import csv
from graphics.viz_metrics import plot_rewards

def plot_from_csv(csv_path):
    # Suppose you saved rewards in a CSV with columns: episode, reward
    rewards = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row['reward']))

    plot_rewards(rewards, window_name="Episode Rewards from CSV")

if __name__ == "__main__":
    csv_file = "./logs/my_rewards.csv"
    plot_from_csv(csv_file)