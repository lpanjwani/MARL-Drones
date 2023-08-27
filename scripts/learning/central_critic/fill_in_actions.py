import numpy as np
import gym
from gym.spaces import Box

from ray.rllib.agents.callbacks import DefaultCallbacks

from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch

ACTION_VECTOR_SIZE = 4


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
        )  # Bounded
        to_update[:, -ACTION_VECTOR_SIZE:] = opponent_actions
