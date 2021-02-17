from typing import (
    Optional,
    Tuple,
    Sequence,
    Union,
    Dict,
)

import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    DistributionType,
    LinearActorCriticHead,
)
from allenact.algorithms.onpolicy_sync.policy import (
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.model_utils import simple_conv_and_linear_weights_init
from torch import Tensor


class RearrangeActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    """A CNN->RNN actor-critic model for rearrangement tasks."""

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        """Initialize a `RearrangeActorCriticSimpleConvRNN` object.

        # Parameters
        action_space : The action space of the agent.
            Should equal `gym.spaces.Discrete(# actions available to the agent)`.
        observation_space : The observation space available to the agent.
        rgb_uuid : The unique id of the RGB image sensor (see `RGBSensor`).
        unshuffled_rgb_uuid : The unique id of the `UnshuffledRGBRearrangeSensor` available to the agent.
        hidden_size : The size of the hidden layer of the RNN.
        num_rnn_layers: The number of hidden layers in the RNN.
        rnn_type : The RNN type, should be "GRU" or "LSTM".
        """
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid
        self.unshuffled_rgb_uuid = unshuffled_rgb_uuid

        self.concat_rgb_uuid = "concat_rgb"
        assert self.concat_rgb_uuid not in observation_space

        self.visual_encoder = self._create_visual_encoder()

        self.state_encoder = RNNStateEncoder(
            self.recurrent_hidden_state_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        self.train()

    def _create_visual_encoder(self) -> nn.Module:
        """Create the visual encoder for the model."""
        img_space: gym.spaces.Box = self.observation_space[self.rgb_uuid]
        return SimpleCNN(
            observation_space=gym.spaces.Dict(
                {
                    self.concat_rgb_uuid: gym.spaces.Box(
                        low=np.tile(img_space.low, (1, 1, 2)),
                        high=np.tile(img_space.high, (1, 1, 2)),
                        shape=img_space.shape[:2] + (img_space.shape[2] * 2,),
                    )
                }
            ),
            output_size=self._hidden_size,
            rgb_uuid=self.concat_rgb_uuid,
            depth_uuid=None,
        )

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
        )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        cur_img = observations[self.rgb_uuid]
        unshuffled_img = observations[self.unshuffled_rgb_uuid]
        concat_img = torch.cat((cur_img, unshuffled_img), dim=-1)

        x = self.visual_encoder({self.concat_rgb_uuid: concat_img})
        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class ResNetRearrangeActorCriticRNN(RearrangeActorCriticSimpleConvRNN):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        """A CNN->RNN rearrangement model that expects ResNet features instead
        of RGB images.

        Nearly identical to `RearrangeActorCriticSimpleConvRNN` but
        `rgb_uuid` should now be the unique id of the ResNetPreprocessor
        used to featurize RGB images using a pretrained ResNet before
        they're passed to this model.
        """
        self.visual_attention: Optional[nn.Module] = None
        super().__init__(**prepare_locals_for_super(locals()))

    def _create_visual_encoder(self) -> nn.Module:
        a, b = [
            self.observation_space[k].shape[0]
            for k in [self.rgb_uuid, self.unshuffled_rgb_uuid]
        ]
        assert a == b
        self.visual_attention = nn.Sequential(
            nn.Conv2d(3 * a, 32, 1,), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1,),
        )
        visual_encoder = nn.Sequential(
            nn.Conv2d(3 * a, self._hidden_size, 1,), nn.ReLU(inplace=True),
        )
        return visual_encoder

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        cur_img_resnet = observations[self.rgb_uuid]
        unshuffled_img_resnet = observations[self.unshuffled_rgb_uuid]
        concat_img = torch.cat(
            (
                cur_img_resnet,
                unshuffled_img_resnet,
                cur_img_resnet * unshuffled_img_resnet,
            ),
            dim=-3,
        )
        batch_shape, features_shape = concat_img.shape[:-3], concat_img.shape[-3:]
        concat_img_reshaped = concat_img.view(-1, *features_shape)
        attention_probs = torch.softmax(
            self.visual_attention(concat_img_reshaped).view(
                concat_img_reshaped.shape[0], -1
            ),
            dim=-1,
        ).view(concat_img_reshaped.shape[0], 1, *concat_img_reshaped.shape[-2:])
        x = (
            (self.visual_encoder(concat_img_reshaped) * attention_probs)
            .mean(-1)
            .mean(-1)
        )
        x = x.view(*batch_shape, -1)

        x, rnn_hidden_states = self.state_encoder(x, memory.tensor("rnn"), masks)

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class TwoPhaseRearrangeActorCriticSimpleConvRNN(ActorCriticModel[CategoricalDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        in_walkthrough_phase_uuid: str,
        is_walkthrough_phase_embedding_dim: int,
        done_action_index: int,
        walkthrougher_should_ignore_action_mask: Optional[Sequence[float]] = None,
        prev_action_embedding_dim: int = 32,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        """A CNN->RNN model for joint training of the Walkthrough and Unshuffle
        tasks.

        Similar to `RearrangeActorCriticSimpleConvRNN` but with some
        additional sensor inputs (e.g. the `InWalkthroughPhaseSensor` is
        used to tell the agent which phase it is in).
        """
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid
        self.unshuffled_rgb_uuid = unshuffled_rgb_uuid
        self.in_walkthrough_phase_uuid = in_walkthrough_phase_uuid

        self.done_action_index = done_action_index

        self.prev_action_embedder = nn.Embedding(
            action_space.n + 1, embedding_dim=prev_action_embedding_dim
        )

        self.is_walkthrough_phase_embedder = nn.Embedding(
            num_embeddings=2, embedding_dim=is_walkthrough_phase_embedding_dim
        )

        self.walkthrough_good_action_logits: Optional[torch.Tensor]
        if walkthrougher_should_ignore_action_mask is not None:
            self.register_buffer(
                "walkthrough_good_action_logits",
                -1000 * torch.FloatTensor(walkthrougher_should_ignore_action_mask),
                persistent=False,
            )
        else:
            self.walkthrough_good_action_logits = None

        self.concat_rgb_uuid = "concat_rgb"
        assert self.concat_rgb_uuid not in observation_space

        self.visual_encoder = self._create_visual_encoder()

        self.state_encoder = RNNStateEncoder(
            prev_action_embedding_dim
            + is_walkthrough_phase_embedding_dim
            + 2 * self.recurrent_hidden_state_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.walkthrough_encoder = RNNStateEncoder(
            self._hidden_size, self._hidden_size, num_layers=1, rnn_type="GRU",
        )

        self.apply(simple_conv_and_linear_weights_init)

        self.walkthrough_ac = LinearActorCriticHead(self._hidden_size, action_space.n)
        self.walkthrough_ac.actor_and_critic.bias.data[self.done_action_index] -= 3
        self.unshuffle_ac = LinearActorCriticHead(self._hidden_size, action_space.n)

        self.train()

    def _create_visual_encoder(self) -> nn.Module:
        img_space: gym.spaces.Box = self.observation_space[self.rgb_uuid]
        return SimpleCNN(
            observation_space=gym.spaces.Dict(
                {
                    self.concat_rgb_uuid: gym.spaces.Box(
                        low=np.tile(img_space.low, (1, 1, 2)),
                        high=np.tile(img_space.high, (1, 1, 2)),
                        shape=img_space.shape[:2] + (img_space.shape[2] * 2,),
                    )
                }
            ),
            output_size=self._hidden_size,
            rgb_uuid=self.concat_rgb_uuid,
            depth_uuid=None,
        )

    def load_state_dict(
        self,
        state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]],
        strict: bool = True,
    ):
        # For backwards compatability, renames "explore" to "walkthrough"
        # in state dict keys.
        for key in list(state_dict.keys()):
            if "explore" in key:
                new_key = key.replace("explore", "walkthrough")
                assert new_key not in state_dict
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        if "walkthrough_good_action_logits" in state_dict:
            del state_dict["walkthrough_good_action_logits"]

        return super(TwoPhaseRearrangeActorCriticSimpleConvRNN, self).load_state_dict(
            state_dict=state_dict, strict=strict
        )

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return dict(
            rnn=(
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            walkthrough_encoding=(
                (
                    ("layer", self.walkthrough_encoder.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
        )

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        in_walkthrough_phase_mask = observations[self.in_walkthrough_phase_uuid]
        in_unshuffle_phase_mask = ~in_walkthrough_phase_mask
        in_walkthrough_float = in_walkthrough_phase_mask.float()
        in_unshuffle_float = in_unshuffle_phase_mask.float()

        # Don't reset hidden state at start of the unshuffle task
        masks_no_unshuffle_reset = (masks.bool() | in_unshuffle_phase_mask).float()

        cur_img = observations[self.rgb_uuid]
        unshuffled_img = observations[self.unshuffled_rgb_uuid]
        concat_img = torch.cat((cur_img, unshuffled_img), dim=-1)

        vis_features = self.visual_encoder({self.concat_rgb_uuid: concat_img})
        prev_action_embeddings = self.prev_action_embedder(
            ((~masks.bool()).long() * (prev_actions.unsqueeze(-1) + 1))
        ).squeeze(-2)
        is_walkthrough_phase_embedding = self.is_walkthrough_phase_embedder(
            in_walkthrough_phase_mask.long()
        ).squeeze(-2)

        to_cat = [
            vis_features,
            prev_action_embeddings,
            is_walkthrough_phase_embedding,
        ]

        rnn_hidden_states = memory.tensor("rnn")
        rnn_outs = []
        obs_for_rnn = torch.cat(to_cat, dim=-1)
        last_walkthrough_encoding = memory.tensor("walkthrough_encoding")
        walkthrough_encoding: Optional[torch.Tensor] = None
        for step in range(masks.shape[0]):
            rnn_out, rnn_hidden_states = self.state_encoder(
                torch.cat(
                    (obs_for_rnn[step : step + 1], last_walkthrough_encoding), dim=-1
                ),
                rnn_hidden_states,
                masks[step : step + 1],
            )
            rnn_outs.append(rnn_out)

            walkthrough_encoding, _ = self.walkthrough_encoder(
                rnn_out,
                last_walkthrough_encoding,
                masks_no_unshuffle_reset[step : step + 1],
            )
            walkthrough_encoding = (
                last_walkthrough_encoding * in_unshuffle_float[step : step + 1]
                + walkthrough_encoding * in_walkthrough_float[step : step + 1]
            )

        memory = memory.set_tensor("walkthrough_encoding", walkthrough_encoding)

        rnn_out = torch.cat(rnn_outs, dim=0)
        walkthrough_dist, walkthrough_vals = self.walkthrough_ac(rnn_out)
        unshuffle_dist, unshuffle_vals = self.unshuffle_ac(rnn_out)

        assert len(in_walkthrough_float.shape) == len(walkthrough_dist.logits.shape)

        if self.walkthrough_good_action_logits is not None:
            walkthrough_logits = (
                walkthrough_dist.logits
                + self.walkthrough_good_action_logits.view(
                    *((1,) * (len(walkthrough_dist.logits.shape) - 1)), -1
                )
            )
        else:
            walkthrough_logits = walkthrough_dist.logits

        actor = CategoricalDistr(
            logits=in_walkthrough_float * walkthrough_logits
            + in_unshuffle_float * unshuffle_dist.logits
        )
        values = (
            in_walkthrough_float * walkthrough_vals
            + in_unshuffle_float * unshuffle_vals
        )

        ac_output = ActorCriticOutput(distributions=actor, values=values, extras={})

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class ResNetTwoPhaseRearrangeActorCriticRNN(TwoPhaseRearrangeActorCriticSimpleConvRNN):
    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        in_walkthrough_phase_uuid: str,
        is_walkthrough_phase_embedding_dim: int,
        done_action_index: int,
        walkthrougher_should_ignore_action_mask: Optional[Sequence[float]] = None,
        prev_action_embedding_dim: int = 32,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        """A CNN->RNN for joint training of the Walkthrough and Unshuffle tasks
        that expects ResNet features instead of RGB images.

        Nearly identical to `TwoPhaseRearrangeActorCriticSimpleConvRNN`
        but `rgb_uuid` should now be the unique id of the
        ResNetPreprocessor used to featurize RGB images using a
        pretrained ResNet before they're passed to this model.
        """
        self.visual_attention: Optional[nn.Module] = None
        super().__init__(**prepare_locals_for_super(locals()))

    def _create_visual_encoder(self) -> nn.Module:
        a, b = [
            self.observation_space[k].shape[0]
            for k in [self.rgb_uuid, self.unshuffled_rgb_uuid]
        ]
        assert a == b
        self.visual_attention = nn.Sequential(
            nn.Conv2d(3 * a, 32, 1,), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1,),
        )
        visual_encoder = nn.Sequential(
            nn.Conv2d(3 * a, self._hidden_size, 1,), nn.ReLU(inplace=True),
        )
        return visual_encoder

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        in_walkthrough_phase_mask = observations[self.in_walkthrough_phase_uuid]
        in_unshuffle_phase_mask = ~in_walkthrough_phase_mask
        in_walkthrough_float = in_walkthrough_phase_mask.float()
        in_unshuffle_float = in_unshuffle_phase_mask.float()

        # Don't reset hidden state at start of the unshuffle task
        masks_no_unshuffle_reset = (masks.bool() | in_unshuffle_phase_mask).float()

        # Visual features
        cur_img_resnet = observations[self.rgb_uuid]
        unshuffled_img_resnet = observations[self.unshuffled_rgb_uuid]
        concat_img = torch.cat(
            (
                cur_img_resnet,
                unshuffled_img_resnet,
                cur_img_resnet * unshuffled_img_resnet,
            ),
            dim=-3,
        )
        batch_shape, features_shape = concat_img.shape[:-3], concat_img.shape[-3:]
        concat_img_reshaped = concat_img.view(-1, *features_shape)
        attention_probs = torch.softmax(
            self.visual_attention(concat_img_reshaped).view(
                concat_img_reshaped.shape[0], -1
            ),
            dim=-1,
        ).view(concat_img_reshaped.shape[0], 1, *concat_img_reshaped.shape[-2:])
        vis_features = (
            (self.visual_encoder(concat_img_reshaped) * attention_probs)
            .mean(-1)
            .mean(-1)
        )
        vis_features = vis_features.view(*batch_shape, -1)

        # Various embeddings
        prev_action_embeddings = self.prev_action_embedder(
            ((~masks.bool()).long() * (prev_actions.unsqueeze(-1) + 1))
        ).squeeze(-2)
        is_walkthrough_phase_embedding = self.is_walkthrough_phase_embedder(
            in_walkthrough_phase_mask.long()
        ).squeeze(-2)

        to_cat = [
            vis_features,
            prev_action_embeddings,
            is_walkthrough_phase_embedding,
        ]

        rnn_hidden_states = memory.tensor("rnn")
        rnn_outs = []
        obs_for_rnn = torch.cat(to_cat, dim=-1)
        last_walkthrough_encoding = memory.tensor("walkthrough_encoding")
        walkthrough_encoding: Optional[torch.Tensor] = None
        for step in range(masks.shape[0]):
            rnn_out, rnn_hidden_states = self.state_encoder(
                torch.cat(
                    (obs_for_rnn[step : step + 1], last_walkthrough_encoding), dim=-1
                ),
                rnn_hidden_states,
                masks[step : step + 1],
            )
            rnn_outs.append(rnn_out)

            walkthrough_encoding, _ = self.walkthrough_encoder(
                rnn_out,
                last_walkthrough_encoding,
                masks_no_unshuffle_reset[step : step + 1],
            )
            walkthrough_encoding = (
                last_walkthrough_encoding * in_unshuffle_float[step : step + 1]
                + walkthrough_encoding * in_walkthrough_float[step : step + 1]
            )

        memory = memory.set_tensor("walkthrough_encoding", walkthrough_encoding)

        rnn_out = torch.cat(rnn_outs, dim=0)
        walkthrough_dist, walkthrough_vals = self.walkthrough_ac(rnn_out)
        unshuffle_dist, unshuffle_vals = self.unshuffle_ac(rnn_out)

        assert len(in_walkthrough_float.shape) == len(walkthrough_dist.logits.shape)

        if self.walkthrough_good_action_logits is not None:
            walkthrough_logits = (
                walkthrough_dist.logits
                + self.walkthrough_good_action_logits.view(
                    *((1,) * (len(walkthrough_dist.logits.shape) - 1)), -1
                )
            )
        else:
            walkthrough_logits = walkthrough_dist.logits

        actor = CategoricalDistr(
            logits=in_walkthrough_float * walkthrough_logits
            + in_unshuffle_float * unshuffle_dist.logits
        )
        values = (
            in_walkthrough_float * walkthrough_vals
            + in_unshuffle_float * unshuffle_vals
        )

        ac_output = ActorCriticOutput(distributions=actor, values=values, extras={})

        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)
