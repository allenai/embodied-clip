from typing import Dict, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F

from habitat.config import Config
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, Policy


@baseline_registry.register_policy
class ObjGoalResNetPolicy(Policy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        **kwargs
    ):
        assert ObjectGoalSensor.cls_uuid in observation_space.spaces
        assert "rgb" in observation_space.spaces or "depth" in observation_space.spaces
        
        super().__init__(
            ObjGoalResNetNet(
                observation_space=observation_space,
                action_space=action_space,
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: spaces.Dict, action_space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs="rgb" in observation_space.spaces,
        )


class ResnetTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        resnet_tensor_shape = (2048, 7, 7),
        class_dims = 32,
        resnet_compressor_hidden_out_dims = (128, 32),
        combiner_hidden_out_dims = (128, 32),
    ):
        super().__init__()
        self.resnet_tensor_shape = resnet_tensor_shape
        self.class_dims = class_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.resnet_compressor = nn.Sequential(
            nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
            nn.ReLU(),
            nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
            nn.ReLU(),
        )
        self.target_obs_combiner = nn.Sequential(
            nn.Conv2d(
                self.resnet_hid_out_dims[1] + self.class_dims,
                self.combine_hid_out_dims[0],
                1,
            ),
            nn.ReLU(),
            nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
        )

    @property
    def output_dims(self):
        return (
            self.combine_hid_out_dims[-1]
            * self.resnet_tensor_shape[1]
            * self.resnet_tensor_shape[2]
        )

    def adapt_input(self, resnet_input):
        nstep, nsampler = resnet_input.shape[:2]
        resnet_input = resnet_input.view(-1, *resnet_input.shape[-3:])
        return resnet_input, nstep, nsampler

    def distribute_target(self, target_emb):
        return target_emb.view(-1, self.class_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def forward(self, resnet_input, target_emb):
        resnet_input, nstep, nsampler = self.adapt_input(resnet_input)
        embs = [
            self.resnet_compressor(resnet_input),
            self.distribute_target(target_emb),
        ]
        x = self.target_obs_combiner(torch.cat(embs, dim=1,))
        x = x.reshape(x.size(0), -1)  # flatten

        return x

class ObjGoalResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
    ):
        super().__init__()

        self._hidden_size = hidden_size
        self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)

        class_dims = 32
        self._n_object_categories = (
            int(observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]) + 1
        )
        self.obj_categories_embedding = nn.Embedding(self._n_object_categories, class_dims)

        self.visual_encoder = ResNetEncoder(
            observation_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        self.goal_encoder = ResnetTensorGoalEncoder(
            resnet_tensor_shape = self.visual_encoder.output_shape,
            class_dims = class_dims,
        )

        self._n_prev_action = 32
        rnn_input_size = self.goal_encoder.output_dims + self._n_prev_action

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def is_blind(self):
        return False

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []

        if "visual_features" in observations:
            visual_feats = observations["visual_features"]
        else:
            visual_feats = self.visual_encoder(observations)

        object_goal = observations[ObjectGoalSensor.cls_uuid].long()
        target_emb = self.obj_categories_embedding(object_goal)
        goal_encoded_feats = self.goal_encoder(visual_feats, target_emb)
        x.append(goal_encoded_feats)

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )

        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks
        )

        return out, rnn_hidden_states
