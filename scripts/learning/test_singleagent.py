"""Test script for single agent problems.

This scripts runs the best model found by one of the executions of `singleagent.py`

Example
-------
To run the script, type in a terminal:

    $ python test_singleagent.py --exp ./results/save-<env>-<algo>-<obs>-<act>-<time_date>

"""
import os
import time
from datetime import datetime
import argparse
import numpy as np
import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)
from gym_pybullet_drones.utils.utils import sync, str2bool

import shared_constants

DEFAULT_GUI = False
DEFAULT_PLOT = True
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_RECORD_VIDEO = True
DEFAULT_COLAB = False


def run(
    exp,
    gui=DEFAULT_GUI,
    plot=DEFAULT_PLOT,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    record_video=DEFAULT_RECORD_VIDEO,
    colab=DEFAULT_COLAB,
):
    #### Load the model from file ##############################
    algo = exp.split("-")[2]

    if os.path.isfile(exp + "/success_model.zip"):
        path = exp + "/success_model.zip"
    elif os.path.isfile(exp + "/best_model.zip"):
        path = exp + "/best_model.zip"
    else:
        print("[ERROR]: no model under the specified path", exp)

    if algo == "ppo":
        model = PPO.load(path)
    if algo == "ddpg":
        model = DDPG.load(path)

    #### Parameters to recreate the environment ################
    env_name = exp.split("-")[1] + "-aviary-v0"
    OBS = ObservationType.KIN if exp.split("-")[3] == "kin" else ObservationType.RGB

    # Parse ActionType instance from file name
    action_name = exp.split("-")[4]
    ACT = [action for action in ActionType if action.value == action_name]
    if len(ACT) != 1:
        raise AssertionError(
            "Result file could have gotten corrupted. Extracted action type does not match any of the existing ones."
        )
    ACT = ACT.pop()

    #### Evaluate the model ####################################
    eval_env = gym.make(
        env_name, aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS, obs=OBS, act=ACT
    )
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    #### Show, record a video, and log the model's performance #
    test_env = gym.make(
        env_name,
        gui=gui,
        record=record_video,
        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
        obs=OBS,
        act=ACT,
    )
    logger = Logger(
        logging_freq_hz=int(
            test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS,
        ),
        num_drones=1,
        output_folder=output_folder,
        colab=colab,
    )
    obs = test_env.reset()
    start = time.time()
    for i in range(6 * int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS)):  # Up to 6''
        action, _states = model.predict(
            obs, deterministic=True  # OPTIONAL 'deterministic=False'
        )
        obs, reward, done, info = test_env.step(action)
        test_env.render()
        if OBS == ObservationType.KIN:
            logger.log(
                drone=0,
                timestamp=i / test_env.SIM_FREQ,
                state=np.hstack(
                    [obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]
                ),
                control=np.zeros(12),
            )
        sync(np.floor(i * test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
    test_env.close()
    logger.save_as_csv("sa")
    if plot:
        logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Single agent reinforcement learning example script using TakeoffAviary"
    )
    parser.add_argument(
        "--exp",
        type=str,
        help="The experiment folder written as ./results/save-<env>-<algo>-<obs>-<act>-<time_date>",
        metavar="",
    )
    parser.add_argument(
        "--gui",
        default=DEFAULT_GUI,
        type=str2bool,
        help="Whether to use PyBullet GUI (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--plot",
        default=DEFAULT_PLOT,
        type=str2bool,
        help="Whether to plot the simulation results (default: True)",
        metavar="",
    )
    parser.add_argument(
        "--output_folder",
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")',
        metavar="",
    )
    parser.add_argument(
        "--record_video",
        default=False,
        type=str2bool,
        help="Whether to record a video (default: False)",
        metavar="",
    )
    parser.add_argument(
        "--colab",
        default=DEFAULT_COLAB,
        type=bool,
        help='Whether example is being run by a notebook (default: "False")',
        metavar="",
    )
    ARGS = parser.parse_args()

    run(**vars(ARGS))
