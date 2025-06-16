import os
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_tensorboard_scalars(logdir, scalar_tags=None):
    scalar_data = {}

    for subdir, dirs, files in os.walk(logdir):
        for file in files:
            if "tfevents" in file:
                filepath = os.path.join(subdir, file)
                for e in tf.compat.v1.train.summary_iterator(filepath):
                    for v in e.summary.value:
                        if scalar_tags is None:
                            if v.tag not in ['rollout/ep_rew_mean', 'train/value_loss', 'train/policy_loss', 'train/entropy_loss', 'time/fps']:
                                continue
                        else:
                            if v.tag not in scalar_tags:
                                continue
                        if v.tag not in scalar_data:
                            scalar_data[v.tag] = []
                        scalar_data[v.tag].append((e.step, v.simple_value))

    # Plotting
    for tag, values in scalar_data.items():
        steps, vals = zip(*values)
        plt.plot(steps, vals, label=tag)

    plt.xlabel("Timesteps")
    plt.ylabel("Metric Value")
    plt.title("Training Metrics from TensorBoard")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
plot_tensorboard_scalars("C:/Users/ereng/PycharmProjects/RLParkingSystemProject/scripts/logs/tensorboard")
