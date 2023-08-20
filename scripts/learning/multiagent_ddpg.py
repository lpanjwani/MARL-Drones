import os
import argparse
from datetime import datetime
import numpy as np
import pybullet as p
import gym
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray
from ray import tune
from ray.tune import register_env
from ray.rllib.agents import ddpg
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

from gym_pybullet_drones.utils.utils import str2bool
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import (
    LeaderFollowerAviary,
)
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import (
    ActionType,
    ObservationType,
)

import shared_constants

OWN_OBSERVATION_VEC_SIZE = 12
ACTION_VECTOR_SIZE = None


class CentralizedCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.action_model = FullyConnectedNetwork(
            Box(low=-1, high=1, shape=(OWN_OBSERVATION_VEC_SIZE,)),
            action_space,
            num_outputs,
            model_config,
            name + "_action",
        )
        self.value_model = FullyConnectedNetwork(
            obs_space, action_space, 1, model_config, name + "_vf"
        )
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        value_out, _ = self.value_model(
            {"obs": self._model_in[0]}, self._model_in[1], self._model_in[2]
        )
        return torch.reshape(value_out, [-1])


class FillInActions(DefaultCallbacks):
    def on_postprocess_trajectory(
        self,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch,
        original_batches,
        **kwargs
    ):
        to_update = postprocessed_batch[SampleBatch.CUR_OBS]
        other_id = 1 if agent_id == 0 else 0
        action_encoder = ModelCatalog.get_preprocessor_for_space(
            Box(-1, 1, (ACTION_VECTOR_SIZE,), np.float32)  # Bounded
        )
        _, opponent_batch = original_batches[other_id]
        opponent_actions = np.array(
            [
                action_encoder.transform(np.clip(a, -1, 1))
                for a in opponent_batch[SampleBatch.ACTIONS]
            ]
        )
        to_update[:, -ACTION_VECTOR_SIZE:] = opponent_actions


def central_critic_observer(agent_obs, **kw):
    new_obs = {
        0: {
            "own_obs": agent_obs[0],
            "opponent_obs": agent_obs[1],
            "opponent_action": np.zeros(ACTION_VECTOR_SIZE),
        },
        1: {
            "own_obs": agent_obs[1],
            "opponent_obs": agent_obs[0],
            "opponent_action": np.zeros(ACTION_VECTOR_SIZE),
        },
    }
    return new_obs


class MultiAgentDDPG:
    action_space = None
    args = None
    environment = None
    environment_name = None
    observer_space = None
    results_directory = None
    tuner_config = None
    tuner_results = None
    tuner_stopper = None

    def __init__(self):
        self.args = self.parse_cli_arguments()
        self.results_directory = self.create_results_directory(self.args)
        self.build_action_vector_size(self.args)
        self.init_ray()
        self.register_spaces(self.args)
        self.build_tuner_config(self.args)
        self.build_tuner_stop_conditions()
        self.run_tuner(self.results_directory, self.tuner_config, self.tuner_stopper)
        self.check_learning_achieved(self.tuner_results)
        self.build_path_file(self.results_directory, self.tuner_results)
        self.shutdown_ray()

    # Parse CLI arguments
    def parse_cli_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num_drones",
            default=2,
            type=int,
            help="Number of drones (default: 2)",
            metavar="",
        )
        parser.add_argument(
            "--env",
            default="leaderfollower",
            type=str,
            choices=["leaderfollower", "flock", "meetup"],
            help="Task (default: leaderfollower)",
            metavar="",
        )
        parser.add_argument(
            "--act",
            default=ActionType.RPM,
            type=ActionType,
            help="Action space (default: RPM)",
            metavar="",
        )
        parser.add_argument(
            "--algo",
            default="cc",
            type=str,
            choices=["cc"],
            help="MARL approach (default: cc)",
            metavar="",
        )
        parser.add_argument(
            "--workers",
            default=0,
            type=int,
            help="Number of RLlib workers (default: 0)",
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

        # Fix to KIN for now
        ARGS.obs = ObservationType.KIN

        return ARGS

    # Create results directory with timestamp
    def create_results_directory(self, ARGS):
        filename = (
            os.path.dirname(os.path.abspath(__file__))
            + "/results/save-"
            + ARGS.env
            + "-"
            + str(ARGS.num_drones)
            + "-"
            + ARGS.algo
            + "-"
            + ARGS.obs.value
            + "-"
            + ARGS.act.value
            + "-"
            + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        )
        if not os.path.exists(filename):
            os.makedirs(filename + "/")

        return filename

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

    def register_gym_env(self, ARGS):
        # Register the custom centralized critic model
        ModelCatalog.register_custom_model("central_a2c_model", CentralizedCriticModel)

        if ARGS.env == "flock":
            self.register_flock_environment(ARGS)
        elif ARGS.env == "leaderfollower":
            self.register_leaderfollower_environment(ARGS)
        elif ARGS.env == "meetup":
            self.register_meetup_environment(ARGS)
        else:
            print("[ERROR] environment not yet implemented")
            exit()

    def register_flock_environment(self, ARGS):
        self.environment_name = "flock-aviary-v0"

        register_env(
            self.environment_name,
            lambda _: FlockAviary(
                num_drones=ARGS.num_drones,
                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                obs=ARGS.obs,
                act=ARGS.act,
            ),
        )

        self.environment = FlockAviary(
            num_drones=ARGS.num_drones,
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
                num_drones=ARGS.num_drones,
                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                obs=ARGS.obs,
                act=ARGS.act,
            ),
        )

        self.environment = LeaderFollowerAviary(
            num_drones=ARGS.num_drones,
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
                num_drones=ARGS.num_drones,
                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                obs=ARGS.obs,
                act=ARGS.act,
            ),
        )

        self.environment = MeetupAviary(
            num_drones=ARGS.num_drones,
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

    def build_tuner_config(self, ARGS):
        self.tuner_config = ddpg.DEFAULT_CONFIG.copy()

        self.tuner_config = {
            "env": self.environment_name,
            "num_workers": 0 + ARGS.workers,
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            "batch_mode": "complete_episodes",
            "callbacks": FillInActions,
            "framework": "torch",
        }

        self.tuner_config["model"] = {
            "custom_model": "central_a2c_model",
        }

        self.tuner_config["multiagent"] = {
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

    def build_tuner_stop_conditions(self):
        self.tuner_stopper = {
            "timesteps_total": 120000,
            # "episode_reward_mean": 0,
            # "training_iteration": 0,
        }

    def run_tuner(self, folder, config, stop):
        self.tuner_results = tune.run(
            DDPGTrainer,
            checkpoint_at_end=True,
            config=config,
            local_dir=folder,
            stop=stop,
            verbose=True,
        )

    def check_learning_achieved(self, results):
        check_learning_achieved(results, 1.0)

    def build_path_file(self, folder, results):
        checkpoints = results.get_trial_checkpoints_paths(
            trial=results.get_best_trial("episode_reward_mean", mode="max"),
            metric="episode_reward_mean",
        )
        with open(folder + "/path.txt", "w+") as f:
            f.write(checkpoints[0][0])


if __name__ == "__main__":
    MultiAgentDDPG()
