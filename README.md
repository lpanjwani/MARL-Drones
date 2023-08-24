### Multi-Agent Reinforcement Learning Drones

## Introduction

This project is a multi-agent reinforcement learning project that uses [MAPPO](https://arxiv.org/abs/2009.09346) and [MADDPG](https://arxiv.org/abs/1706.02275) algorithms to train a group of drones. The drones are trained in a simulated environment adopted from [Gym-PyBullet-Drones](https://github.com/utiasDSL/gym-pybullet-drones).

## Dependencies

-   [PyBullet](https://github.com/bulletphysics/bullet3)
-   [OpenAI Gym](https://github.com/openai/gym)
-   [RLib/Ray](https://github.com/ray-project/ray)
-   [PyTorch](https://pytorch.org/)
-   [NumPy](https://numpy.org/)
-   [Matplotlib](https://matplotlib.org/)

## Installation

```bash
python3 -m pip install -e .
```

## Usage

### Single Agent

#### Policy Proximal Optimization (PPO)

```bash
gui=False
record_video=True
n_gpu = 1
algo_type = "hover"
action_type = "pid"

python scripts/learning/singleagent_ppo.py \
  --env=$algo_type \
  --act=$action_type \
  --record_video=$record_video \
  --gui=$gui
```

#### Deep Deterministic Policy Gradient (DDPG)

```bash
gui=False
record_video=True
n_gpu = 1
algo_type = "hover"
action_type = "pid"

python scripts/learning/singleagent_ddpg.py \
  --env=$algo_type \
  --act=$action_type \
  --record_video=$record_video \
  --gui=$gui
```

### Multi-Agent

#### Policy Proximal Optimization (PPO)

For Training:

```bash
gui=False
record_video=True
num_drones = 2
n_gpu = 0
algo_type = "flock"

RLLIB_NUM_GPUS=$n_gpu \
  python scripts/learning/multiagent_ppo.py \
  --env $algo_type \
  --num_drones $num_drones \
  --record_video=$record_video \
  --gui=$gui
```

For Testing:

```bash
colab=True
gui=False
record_video=True
result_folder="./results/save-flock-2-cc-kin-rpm-08.21.2023_13.55.36"

python scripts/learning/test_multiagent_ppo.py \
  --exp $result_folder \
  --record_video=$record_video \
  --gui=$gui \
   --colab=$colab
```

#### Deep Deterministic Policy Gradient (DDPG)

For Training:

```bash
gui=False
record_video=True
num_drones = 2
n_gpu = 0
algo_type = "flock"

RLLIB_NUM_GPUS=$n_gpu \
  python scripts/learning/multiagent_ddpg.py \
  --env $algo_type \
  --num_drones $num_drones \
  --record_video=$record_video \
  --gui=$gui
```

For Testing:

```bash
colab=True
gui=False
record_video=True
result_folder="./results/save-flock-2-cc-kin-rpm-08.21.2023_13.55.36"

python scripts/learning/test_multiagent_ddpg.py \
  --exp $result_folder \
  --record_video=$record_video \
  --gui=$gui \
   --colab=$colab
```

## Results

```bash
csv =  "./results/save-flock-2-cc-kin-rpm-08.21.2023_13.55.36/DDPG/progress.csv"
label = "Multi-Agent DDPG"

python scripts/plots/results.py \
  --csv $csv \
  --label $label
```
