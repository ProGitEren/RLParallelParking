from stable_baselines3.common.callbacks import BaseCallback
import torch
import matplotlib.pyplot as plt
import os

class FeatureLoggingCallback(BaseCallback):
    def __init__(self, log_dir="logs/features", verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % 500 == 0:  # Log every 500 steps
            obs_tensor = torch.tensor(self.locals["new_obs"], device=self.model.device)

            with torch.no_grad():
                _ = self.model.policy.extract_features(obs_tensor)

            # Extract features
            cnn_feat = self.model.policy.features_extractor.last_cnn_out[0].cpu().numpy()  # (batch, 3328)
            cnn_feat_reshaped = cnn_feat.reshape(32, 13, 8)  # 32 channels

            # Plot and save all 32 channels
            fig, axes = plt.subplots(4, 8, figsize=(16, 8))
            for i, ax in enumerate(axes.flat):
                ax.imshow(cnn_feat_reshaped[i], cmap="viridis")
                ax.set_title(f"Channel {i}")
                ax.axis("off")

            plt.tight_layout()
            save_path = os.path.join(self.log_dir, f"cnn_features_step_{self.num_timesteps}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"[FEATURE LOG] Saved CNN feature map at: {save_path}")

        return True
