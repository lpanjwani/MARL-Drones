import os
import time
import argparse
import numpy as np
import gym
from gym.spaces import Dict
import ray
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog

from scripts.learning.central_a2c.model import CentralizedCriticModel
from scripts.learning.central_a2c.fill_in_actions import FillInActions
from scripts.learning.central_a2c.observer import (
    central_critic_observer,
)

from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import (
    LeaderFollowerAviary,
)
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
)
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

import shared_constants

OWN_OBSERVATION_VEC_SIZE = 12
ACTION_VECTOR_SIZE = 4

DEFAULT_COLAB = False


class MultiAgentPPOTester:
    action = None
    action_name = None
    action_space = None
    args = None
    environment = None
    environment_name = None
    num_drones = None
    observer_space = None
    trainer = None
    trainer_config = None

    def __init__(self):
        self.args = self.parse_cli_arguments()
        self.parse_arguments_pattern(self.args)
        self.build_action_vector_size(self.args)
        self.register_gym_environment(self.args)
        self.register_spaces()
        self.build_config(self.args)
        self.restore_trainer()
        self.test_performance()
        self.shutdown_ray()

    # Parse CLI arguments
    def parse_cli_arguments(self):
        parser = argparse.ArgumentParser()
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

        return ARGS

    def parse_arguments_pattern(self, ARGS):
        self.num_drones = int(ARGS.exp.split("-")[2])

        self.action_name = ARGS.exp.split("-")[5]
        self.action = [
            action for action in ActionType if action.value == self.action_name
        ]
        self.action = self.action.pop()

    # Build action constants
    def build_action_vector_size(self, ARGS):
        if ARGS.act in [
            ActionType.ONE_D_RPM,
            ActionType.ONE_D_DYN,
            ActionType.ONE_D_PID,
        ]:
            ACTION_VECTOR_SIZE = 1
        elif ARGS.act in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
            ACTION_VECTOR_SIZE = 4
        elif ARGS.act == ActionType.PID:
            ACTION_VECTOR_SIZE = 3
        else:
            print("[ERROR] unknown ActionType")
            exit()

    # Initialize Ray Server
    def init_ray(self):
        self.shutdown_ray()
        ray.init(ignore_reinit_error=True)

    # Shutdown Ray Server
    def shutdown_ray(self):
        ray.shutdown()

    def register_gym_environment(self, ARGS):
        # Register the custom centralized critic model
        ModelCatalog.register_custom_model("central_a2c_model", CentralizedCriticModel)

        if ARGS.env == "flock":
            return self.register_flock_environment(ARGS)
        elif ARGS.env == "leaderfollower":
            return self.register_leaderfollower_environment(ARGS)
        elif ARGS.env == "meetup":
            return self.register_meetup_environment(ARGS)
        else:
            print("[ERROR] environment not yet implemented")
            exit()

    def register_flock_environment(self, ARGS):
        self.environment_name = "flock-aviary-v0"

        register_env(
            self.environment_name,
            lambda _: FlockAviary(
                num_drones=self.num_drones,
                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                obs=ARGS.obs,
                act=ARGS.act,
            ),
        )

        self.environment = FlockAviary(
            num_drones=self.num_drones,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=ARGS.obs,
            act=ARGS.act,
            gui=ARGS.gui,
            record=ARGS.record_video,
        )

    def register_leaderfollower_environment(self, ARGS):
        self.environment_name = "leaderfollower-aviary-v0"

        register_env(
            self.environment_name,
            lambda _: LeaderFollowerAviary(
                num_drones=self.num_drones,
                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                obs=ARGS.obs,
                act=ARGS.act,
            ),
        )

        self.environment = LeaderFollowerAviary(
            num_drones=self.num_drones,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=ARGS.obs,
            act=ARGS.act,
            gui=ARGS.gui,
            record=ARGS.record_video,
        )

    def register_meetup_environment(self, ARGS):
        self.environment_name = "meetup-aviary-v0"

        register_env(
            self.environment_name,
            lambda _: MeetupAviary(
                num_drones=self.num_drones,
                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                obs=ARGS.obs,
                act=ARGS.act,
            ),
        )

        self.environment = MeetupAviary(
            num_drones=self.num_drones,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=ARGS.obs,
            act=ARGS.act,
            gui=ARGS.gui,
            record=ARGS.record_video,
        )

    def register_spaces(self):
        self.observer_space = Dict(
            {
                "own_obs": self.environment.observation_space[0],
                "opponent_obs": self.environment.observation_space[0],
                "opponent_action": self.environment.action_space[0],
            }
        )

        self.action_space = self.environment.action_space[0]

    def build_config(self, ARGS):
        self.trainer_config = ppo.DEFAULT_CONFIG.copy()

        self.trainer_config = {
            "env": self.environment_name,
            "num_workers": 0 + ARGS.workers,
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "batch_mode": "complete_episodes",
            "callbacks": FillInActions,
            "framework": "torch",
        }

        self.trainer_config["model"] = {
            "custom_model": "central_a2c_model",
        }

        self.trainer_config["multiagent"] = {
            "policies": {
                "pol0": (
                    None,
                    self.observer_space,
                    self.action_space,
                    {
                        "agent_id": 0,
                    },
                ),
                "pol1": (
                    None,
                    self.observer_space,
                    self.action_space,
                    {
                        "agent_id": 1,
                    },
                ),
            },
            "policy_mapping_fn": lambda x: "pol0" if x == 0 else "pol1",
            "observation_fn": central_critic_observer,
        }

    def restore_trainer(self):
        self.trainer = ppo.PPOTrainer(config=self.trainer_config)

        with open(self.args.exp + "/path.txt", "r+") as f:
            checkpoint = f.read()

        self.trainer.restore(checkpoint)

    def extract_policies(self):
        policy0 = self.trainer.get_policy("pol0")
        print("action model 0", policy0.model.action_model)
        print("value model 0", policy0.model.value_model)

        policy1 = self.trainer.get_policy("pol1")
        print("action model 1", policy1.model.action_model)
        print("value model 1", policy1.model.value_model)

        return policy0, policy1

    def test_performance(self):
        obs = self.environment.reset()

        logger = Logger(
            logging_freq_hz=int(
                self.environment.SIM_FREQ / self.environment.AGGR_PHY_STEPS
            ),
            num_drones=self.num_drones,
            colab=self.args.colab,
        )

        policy0, policy1 = self.extract_policies()

        if self.action in [
            ActionType.ONE_D_RPM,
            ActionType.ONE_D_DYN,
            ActionType.ONE_D_PID,
        ]:
            action = {i: np.array([0]) for i in range(self.num_drones)}
        elif self.action in [ActionType.RPM, ActionType.DYN, ActionType.VEL]:
            action = {i: np.array([0, 0, 0, 0]) for i in range(self.num_drones)}
        elif self.action == ActionType.PID:
            action = {i: np.array([0, 0, 0]) for i in range(self.num_drones)}
        else:
            print("[ERROR] unknown ActionType")
            exit()

        start = time.time()

        for i in range(
            6 * int(self.environment.SIM_FREQ / self.environment.AGGR_PHY_STEPS)
        ):
            temp = {}
            temp[0] = policy0.compute_single_action(
                np.hstack([action[1], obs[1], obs[0]])
            )
            temp[1] = policy1.compute_single_action(
                np.hstack([action[0], obs[0], obs[1]])
            )
            action = {0: temp[0][0], 1: temp[1][0]}

            obs, reward, done, info = self.environment.step(action)

            self.environment.render()

            for j in range(self.num_drones):
                logger.log(
                    drone=j,
                    timestamp=i / self.environment.SIM_FREQ,
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

            sync(
                np.floor(i * self.environment.AGGR_PHY_STEPS),
                start,
                self.environment.TIMESTEP,
            )

        self.environment.close()
        logger.save_as_csv("ma")  # Optional CSV save
        logger.plot()


if __name__ == "__main__":
    MultiAgentPPOTester()
