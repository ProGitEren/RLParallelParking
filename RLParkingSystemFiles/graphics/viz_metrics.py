# graphics/viz_metrics.py

import matplotlib.pyplot as plt

def plot_rewards(reward_history, window_name="Episode Rewards"):
    """
    reward_history: list or array of episode rewards
    """
    plt.figure()
    plt.plot(reward_history, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(window_name)
    plt.legend()
    plt.show(block=False)

def plot_success_rate(success_rates, window_name="Success Rates"):
    """
    success_rates: array of success rates over training epochs
    """
    plt.figure()
    plt.plot(success_rates, label="Success Rate")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Success Rate (%)")
    plt.title(window_name)
    plt.legend()
    plt.show(block=False)