from typing import Tuple, Optional

import gym
import torch
from gym.spaces.dict import Dict as SpaceDict

from allenact.algorithms.onpolicy_sync.policy import (
    ObservationType,
    DistributionType
)
from allenact.base_abstractions.misc import (
    Memory,
    ActorCriticOutput,
)
from allenact.embodiedai.models.visual_nav_models import VisualNavActorCritic

class CLIPZeroshotNavActorCritic(VisualNavActorCritic):
    action_space: gym.spaces.Discrete

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        hidden_size=512,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=False,
            beliefs_fusion=None,
            auxiliary_uuids=None,
        )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:

        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        vis_embeds = self.visual_forward_encoder(observations)

        # 2. use RNNs to get single/multiple beliefs
        belief, rnn_hidden_states = self.state_encoders['single_belief'](
            vis_embeds,
            memory.tensor(key),
            masks
        )
        beliefs_dict = { 'single_belief': belief }
        memory.set_tensor('single_belief', rnn_hidden_states)

        # 3. fuse beliefs for multiple belief models
        beliefs, task_weights = self.fuse_beliefs(beliefs_dict, None)

        goal_embeds = self.goal_forward_encoder(observations)

        print(vis_embeds.shape)
        print(beliefs.shape)
        print(goal_embeds.shape)

        output = (obs_embeds + beliefs) * goal_embeds

        # 4. prepare output
        actor_critic_output = ActorCriticOutput(
            distributions=self.actor(output),
            values=self.critic(output),
            extras={},
        )

        return actor_critic_output, memory

    def visual_forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        raise NotImplementedError("Obs Encoder Not Implemented")

    def goal_forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        raise NotImplementedError("Goal Encoder Not Implemented")


class CLIPZeroshotObjectNavActorCritic(VisualNavActorCritic):
    def __init__(
        # base params
        self,
        action_space: gym.spaces.Discrete,
        observation_space: SpaceDict,
        goal_sensor_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
        add_prev_actions=False,
        add_prev_action_null_token=False,
        action_embed_size=6,
        # custom params
        clip_rgb_preprocessor_uuid: str = 'rgb_clip_resnet',
        clip_text_preprocessor_uuid: str = 'text_clip_resnet',
        clip_embedding_dim: int = 1024
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size
        )

        assert clip_rgb_preprocessor_uuid is not None and clip_text_preprocessor_uuid is not None

        self.clip_rgb_preprocessor_uuid = clip_rgb_preprocessor_uuid
        self.clip_text_preprocessor_uuid = clip_text_preprocessor_uuid

        self.create_state_encoders(
            obs_embed_size=clip_embedding_dim,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=clip_embedding_dim,
            action_embed_size=action_embed_size,
        )

        self.train()

    @property
    def is_blind(self) -> bool:
        return False

    def visual_forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.observations[self.clip_rgb_preprocessor_uuid]

    def goal_forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.observations[self.clip_text_preprocessor_uuid]
