import matplotlib.pyplot as plt
import numpy as np

def plot_training(first_results, second_results=None, window=50, save_prefix="plot", first_prefix='Single-Agent', second_prefix='Multi-Agent'):
    """
        Plot RL training results for multi-agent systems.

        Args:
            first_results (dict): {
                "rewards": [...],    # total reward per episode
                "steps": [...]       # steps per episode
            }
            second_results (dict or None): same structure as first_results
            window (int): smoothing window size for moving average
            save_prefix (str): prefix of graphs
            first_prefix (str): first training results
            second_prefix (str): second training results
    """

    def moving_average(x, w=50):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    # Rewards
    plt.figure(figsize=(10, 6))
    plt.plot(first_results["rewards"], label=f"{first_prefix} (raw)", alpha=0.3)
    plt.plot(moving_average(first_results["rewards"], window), label=f"{first_prefix} (smoothed)", linewidth=2)

    if second_results:
        plt.plot(second_results["rewards"], label=f"{second_prefix} (raw)", alpha=0.3)
        plt.plot(moving_average(second_results["rewards"], window), label=f"{second_prefix} (smoothed)", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward per Episode")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_prefix}_rewards.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Steps
    plt.figure(figsize=(10, 6))
    plt.plot(first_results["steps"], label=f"{first_prefix}", alpha=0.3)
    plt.plot(moving_average(first_results["steps"], window), label=f"{first_prefix} (smoothed)", linewidth=2)
    if second_results:
        plt.plot(second_results["steps"], label=f"{second_prefix}", alpha=0.3)
        plt.plot(moving_average(second_results["steps"], window), label=f"{second_prefix} (smoothed)", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Steps per Episode")
    plt.title("Steps Needed per Episode")
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_prefix}_steps.png", dpi=300, bbox_inches="tight")
    plt.close()
