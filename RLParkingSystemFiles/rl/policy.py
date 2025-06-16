import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gym import spaces


class CustomMLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        super(CustomMLPExtractor, self).__init__(observation_space, features_dim=256)

        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
        )

    def forward(self, observations:th.Tensor) -> th.Tensor:
        return self.net(observations)

class CustomParkingPolicyContinuous(ActorCriticPolicy):
    def __init(self,
               observation_space:spaces.Box,
               action_space:spaces.Box,
               lr_schedule,
               net_arch=None,
               activation_fn=nn.Tanh,
               *args,
               **kwargs
               ):
        super(CustomParkingPolicyContinuous, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch = [dict(pi=(256, 256), vf= [256, 256])],
            activation_fn= activation_fn,
            features_extractor_class=CustomMLPExtractor,
            features_extractor_kwargs=dict(),
            *args,
            **kwargs
        )

class CustomParkingPolicyDiscrete(ActorCriticPolicy):
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 lr_schedule,
                 net_arch=None,
                 activation_fn=nn.Tanh,
                 *args,
                 **kwargs):

        super(CustomParkingPolicyDiscrete, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=activation_fn,
            features_extractor_class=CustomMLPExtractor,
            features_extractor_kwargs={},
            *args,
            **kwargs
        )


class HybridCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=256)

        ogm_dim = 60 * 40
        total_dim = observation_space.shape[0]
        self.ogm_len = ogm_dim
        self.state_len = total_dim - ogm_dim  # e.g., yaw + speed + 6 dist + gear = 9

        self.last_cnn_out = None

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),  # 60x40 → 28x18
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # 28x18 → 13x8
            nn.ReLU(),
            nn.Flatten()
        )

        cnn_output_dim = 32 * 13 * 8

        self.mlp = nn.Sequential(
            nn.Linear(self.state_len, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Linear(cnn_output_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        ogm = obs[:, 2:2 + self.ogm_len]
        state = th.cat([obs[:, :2], obs[:, 2 + self.ogm_len:]], dim=1)

        ogm_img = ogm.view(-1, 1, 60, 40)
        cnn_out = self.cnn(ogm_img)
        self.last_cnn_out = cnn_out.detach()  # <-- THIS LINE stores the features
        mlp_out = self.mlp(state)

        fused = th.cat([cnn_out, mlp_out], dim=1)
        return self.final(fused)


class HybridParkingPolicyDiscrete(ActorCriticPolicy):
    def __init__(self,
                 observation_space: spaces.Box,
                 action_space: spaces.Discrete,
                 lr_schedule,
                 net_arch=None,
                 activation_fn=nn.Tanh,
                 *args,
                 **kwargs):

        super(HybridParkingPolicyDiscrete, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=activation_fn,
            features_extractor_class=HybridCNNExtractor,
            features_extractor_kwargs={},
            *args,
            **kwargs
        )