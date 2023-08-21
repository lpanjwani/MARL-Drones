import os
import time
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
import ray
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import (
    LeaderFollowerAviary,
)
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

import shared_constants

OWN_OBSERVATION_VEC_SIZE = 12
ACTION_VECTOR_SIZE = 4

DEFAULT_COLAB = False


############################################################
def central_critic_observer(agent_obs, **kw):
    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "opponent_action": np.zeros(
                ACTION_VECTOR_SIZE
            ),  # Filled in by FillInActions
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "opponent_action": np.zeros(
                ACTION_VECTOR_SIZE
            ),  # Filled in by FillInActions
        },
    }
    return new_obs


############################################################
if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Multi-agent reinforcement learning experiments script"
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="The experiment folder written as ./results/save-<env>-<num_drones>-<algo>-<obs>-<act>-<time_date>",
        metavar="",
    )
    parser.add_argument(
        "--colab",
        default=DEFAULT_COLAB,
        type=bool,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )
    parser.add_argument(
        "--gui",
        default=True,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=False,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    ARGS = parser.parse_args()

    #### Parameters to recreate the environment ################
    NUM_DRONES = int(ARGS.exp.split("-")[2])
    OBS = (
        ObservationType.KIN if ARGS.exp.split("-")[4] == "kin" else ObservationType.RGB
    )

    # Parse ActionType instance from file name
    action_name = ARGS.exp.split("-")[5]
    ACT = [action for action in ActionType if action.value == action_name]
    if len(ACT) != 1:
        raise AssertionError(
            "Result file could have gotten corrupted. Extracted action type does not match any of the existing ones."
        )
    ACT = ACT.pop()

    #### Constants, and errors #################################
    if OBS == ObservationType.KIN:
        OWN_OBSERVATION_VEC_SIZE = 12
    elif OBS == ObservationType.RGB:
        print("[ERROR] ObservationType.RGB for multi-agent systems not yet implemented")
        exit()
    else:
        print("[ERROR] unknown ObservationType")
        exit()
    if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        ACTION_VECTOR_SIZE = 1
    elif ACT in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        ACTION_VECTOR_SIZE = 4
    elif ACT == ActionType.PID:
        ACTION_VECTOR_SIZE = 3
    else:
        print("[ERROR] unknown ActionType")
        exit()

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the custom centralized critic model ##########
    ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)

    #### Register the environment ##############################
    temp_env_name = "this-aviary-v0"
    if ARGS.exp.split("-")[1] == "flock":
        register_env(
            temp_env_name,
            lambda _: FlockAviary(
                num_drones=NUM_DRONES,
                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                obs=OBS,
                act=ACT,
            ),
        )
    elif ARGS.exp.split("-")[1] == "leaderfollower":
        register_env(
            temp_env_name,
            lambda _: LeaderFollowerAviary(
                num_drones=NUM_DRONES,
                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                obs=OBS,
                act=ACT,
            ),
        )
    elif ARGS.exp.split("-")[1] == "meetup":
        register_env(
            temp_env_name,
            lambda _: MeetupAviary(
                num_drones=NUM_DRONES,
                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                obs=OBS,
                act=ACT,
            ),
        )
    else:
        print("[ERROR] environment not yet implemented")
        exit()

    #### Unused env to extract the act and obs spaces ##########
    if ARGS.exp.split("-")[1] == "flock":
        temp_env = FlockAviary(
            num_drones=NUM_DRONES,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=OBS,
            act=ACT,
        )
    elif ARGS.exp.split("-")[1] == "leaderfollower":
        temp_env = LeaderFollowerAviary(
            num_drones=NUM_DRONES,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=OBS,
            act=ACT,
        )
    elif ARGS.exp.split("-")[1] == "meetup":
        temp_env = MeetupAviary(
            num_drones=NUM_DRONES,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=OBS,
            act=ACT,
        )
    else:
        print("[ERROR] environment not yet implemented")
        exit()
    observer_space = Dict(
        {
            "own_obs": temp_env.observation_space[0],
            "opponent_obs": temp_env.observation_space[0],
            "opponent_action": temp_env.action_space[0],
        }
    )
    action_space = temp_env.action_space[0]

    #### Set up the trainer's config ###########################
    config = (
        ppo.DEFAULT_CONFIG.copy()
    )  # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": temp_env_name,
        "num_workers": 0,  # 0+ARGS.workers,
        "num_gpus": int(
            os.environ.get("RLLIB_NUM_GPUS", "0")
        ),  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        "callbacks": FillInActions,
        "framework": "torch",
    }

    #### Set up the model parameters of the trainer's config ###
    config["model"] = {
        "custom_model": "cc_model",
    }

    #### Set up the multiagent params of the trainer's config ##
    config["multiagent"] = {
        "policies": {
            "pol0": (
                None,
                observer_space,
                action_space,
                {
                    "agent_id": 0,
                },
            ),
            "pol1": (
                None,
                observer_space,
                action_space,
                {
                    "agent_id": 1,
                },
            ),
        },
        "policy_mapping_fn": lambda x: "pol0"
        if x == 0
        else "pol1",  # # Function mapping agent ids to policy ids
        "observation_fn": central_critic_observer,  # See rllib/evaluation/observation_function.py for more info
    }

    #### Restore agent #########################################
    agent = ppo.PPOTrainer(config=config)
    with open(ARGS.exp + "/path.txt", "r+") as f:
        checkpoint = f.read()
    agent.restore(checkpoint)

    #### Extract and print policies ############################
    policy0 = agent.get_policy("pol0")
    print("action model 0", policy0.model.action_model)
    print("value model 0", policy0.model.value_model)
    policy1 = agent.get_policy("pol1")
    print("action model 1", policy1.model.action_model)
    print("value model 1", policy1.model.value_model)

    #### Create test environment ###############################
    if ARGS.exp.split("-")[1] == "flock":
        test_env = FlockAviary(
            num_drones=NUM_DRONES,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=OBS,
            act=ACT,
            gui=ARGS.gui,
            record=ARGS.record_video,
        )
    elif ARGS.exp.split("-")[1] == "leaderfollower":
        test_env = LeaderFollowerAviary(
            num_drones=NUM_DRONES,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=OBS,
            act=ACT,
            gui=ARGS.gui,
            record=ARGS.record_video,
        )
    elif ARGS.exp.split("-")[1] == "meetup":
        test_env = MeetupAviary(
            num_drones=NUM_DRONES,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=OBS,
            act=ACT,
            gui=ARGS.gui,
            record=ARGS.record_video,
        )
    else:
        print("[ERROR] environment not yet implemented")
        exit()

    #### Show, record a video, and log the model's performance #
    obs = test_env.reset()
    logger = Logger(
        logging_freq_hz=int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS),
        num_drones=NUM_DRONES,
        colab=ARGS.colab,
    )
    if ACT in [ActionType.ONE_D_RPM, ActionType.ONE_D_DYN, ActionType.ONE_D_PID]:
        action = {i: np.array([0]) for i in range(NUM_DRONES)}
    elif ACT in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
        action = {i: np.array([0, 0, 0, 0]) for i in range(NUM_DRONES)}
    elif ACT == ActionType.PID:
        action = {i: np.array([0, 0, 0]) for i in range(NUM_DRONES)}
    else:
        print("[ERROR] unknown ActionType")
        exit()
    start = time.time()
    for i in range(6 * int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS)):  # Up to 6''
        #### Deploy the policies ###################################
        temp = {}
        temp[0] = policy0.compute_single_action(
            np.hstack([action[1], obs[1], obs[0]])
        )  # Counterintuitive order, check params.json
        temp[1] = policy1.compute_single_action(np.hstack([action[0], obs[0], obs[1]]))
        action = {0: temp[0][0], 1: temp[1][0]}
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        if OBS == ObservationType.KIN:
            for j in range(NUM_DRONES):
                logger.log(
                    drone=j,
                    timestamp=i / test_env.SIM_FREQ,
                    state=np.hstack(
                        [
                            obs[j][0:3],
                            np.zeros(4),
                            obs[j][3:15],
                            np.resize(action[j], (4)),
                        ]
                    ),
                    control=np.zeros(12),
                )
        sync(np.floor(i * test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
        # if done["__all__"]: obs = test_env.reset() # OPTIONAL EPISODE HALT
    test_env.close()
    logger.save_as_csv("ma")  # Optional CSV save
    logger.plot()

    #### Shut down Ray #########################################
    ray.shutdown()
