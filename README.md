
# 🅿️ RL-Based Parallel Parking with CARLA + PPO

This repository implements a reinforcement learning system for autonomous **parallel parking** using the CARLA simulator and PPO (Proximal Policy Optimization) from Stable-Baselines3.

It supports both continuous and discrete control modes, and includes dynamic scenario generation, occupancy grid mapping using LiDAR, and a variety of sensors for training robust parking policies.

---

## 📦 Project Structure

.
├── envs/ # Custom Gym-like environments built on CARLA
│ └── carla_parallel_env_4.py
│ └── carla_world.py
│
├── rl/ # PPO training algorithms and policy networks
│ └── trainer.py
│ └── policy.py
│
├── sensors/ # Sensor management, LiDAR processing, occupancy mapping
│ └── sensor_manager.py
│ └── data_fusion.py
│
├── scripts/ # Scripts for training and evaluation
│ └── run_training.py
│ └── run_evaluation.py
│
├── graphics/ # Visualization tools for occupancy grids
│ └── viz_occupancy.py
│
├── logs/ # Logs, saved videos, tensorboard, models (auto-generated)
│
├── Debug/ # Full script for single-episode testing and debugging
│
├── docs/ # (Optional) Documentation, diagrams, notes

yaml
Kopyala
Düzenle

---

## 🚗 Features

- ✅ **CARLA World Management** with ego vehicle and NPC boundary vehicles
- ✅ **Custom Parking Environments** (`CarlaParallelParkingEnv`, `HybridEnv`)
- ✅ **LiDAR → Occupancy Grid Mapping** for environment representation
- ✅ **Sensor Manager** (camera, LiDAR, BEV, collision)
- ✅ **PPO with Custom Policies** (MLP and Hybrid CNN-based)
- ✅ **Dynamic Scenario Generation** (parking gap, wall, NPC spawn)
- ✅ **Evaluation & Logging** with success rate and episode returns
- ✅ **Video recording and OGM visualizations**

---

## 🧪 Training

Run the training script for discrete control:

```bash
python scripts/run_training.py
Or run the debug mode for a single-episode full simulation:

bash
Kopyala
Düzenle
python Debug/debug_run_single_episode.py
Logs, videos, and models will be saved under logs/.

🧠 PPO Policy Types
CustomParkingPolicyContinuous: for continuous action training

HybridParkingPolicyDiscrete: for CNN + OGM input with discrete actions

🧾 Requirements
Python 3.8

CARLA Simulator (v0.9.13)

stable-baselines3

gymnasium

torch

opencv-python

numpy

matplotlib

networkx

Install all dependencies with:

bash
Kopyala
Düzenle
pip install -r requirements.txt
📊 Evaluation
Run:

bash
Kopyala
Düzenle
python scripts/run_evaluation.py
Reports reward per episode and success rate.

Uses criteria like no collision and successful alignment to define success.

📍 Notes
Trained and tested on Town03.

BEV and RGB cameras can be toggled.

Designed for headless use on GPU servers (supports xvfb-run).

