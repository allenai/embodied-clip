from typing import (
    Optional,
    Tuple,
    Sequence,
    Union,
    Dict,
    Any,
)

import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

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
from allenact.embodiedai.mapping.mapping_models.active_neural_slam import (
    ActiveNeuralSLAM,
)
from allenact.embodiedai.models.basic_models import SimpleCNN, RNNStateEncoder
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.utils.model_utils import simple_conv_and_linear_weights_init


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
        self.visual_attention.apply(simple_conv_and_linear_weights_init)
        visual_encoder.apply(simple_conv_and_linear_weights_init)

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

        # Various embeddings
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
            last_walkthrough_encoding = (
                last_walkthrough_encoding * in_unshuffle_float[step : step + 1]
                + walkthrough_encoding * in_walkthrough_float[step : step + 1]
            )

        memory = memory.set_tensor("walkthrough_encoding", last_walkthrough_encoding)

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
        self.visual_attention.apply(simple_conv_and_linear_weights_init)
        visual_encoder.apply(simple_conv_and_linear_weights_init)

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
        masks_with_unshuffle_reset = masks.float()
        del masks  # Just to make sure we don't accidentally use `masks when we want `masks_no_unshuffle_reset`

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
            (
                (~masks_with_unshuffle_reset.bool()).long()
                * (prev_actions.unsqueeze(-1) + 1)
            )
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

        for step in range(masks_with_unshuffle_reset.shape[0]):
            rnn_out, rnn_hidden_states = self.state_encoder(
                torch.cat(
                    (
                        obs_for_rnn[step : step + 1],
                        last_walkthrough_encoding
                        * masks_no_unshuffle_reset[step : step + 1],
                    ),
                    dim=-1,
                ),
                rnn_hidden_states,
                masks_with_unshuffle_reset[step : step + 1],
            )
            rnn_outs.append(rnn_out)

            walkthrough_encoding, _ = self.walkthrough_encoder(
                rnn_out,
                last_walkthrough_encoding,
                masks_no_unshuffle_reset[step : step + 1],
            )
            last_walkthrough_encoding = (
                last_walkthrough_encoding * in_unshuffle_float[step : step + 1]
                + walkthrough_encoding * in_walkthrough_float[step : step + 1]
            )

        memory = memory.set_tensor("walkthrough_encoding", last_walkthrough_encoding)

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


class WalkthroughActorCriticResNetWithPassiveMap(RearrangeActorCriticSimpleConvRNN):
    """A CNN->RNN actor-critic model for rearrangement tasks."""

    def __init__(
        self,
        height_map_channels: int,
        semantic_map_channels: int,
        map_kwargs: Dict[str, Any],
        **kwargs
    ):
        super().__init__(**kwargs)

        assert "n_map_channels" not in map_kwargs
        map_kwargs["n_map_channels"] = height_map_channels + semantic_map_channels
        self.height_map_channels = height_map_channels
        self.semantic_map_channels = semantic_map_channels
        self.map = ActiveNeuralSLAM(**map_kwargs)

        self.resnet_features_downsampler = nn.Sequential(
            nn.Conv2d(512, 64, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, 3),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
        )
        self.resnet_features_downsampler.apply(simple_conv_and_linear_weights_init)

        self.resnet_normalizer = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.LayerNorm(normalized_shape=[512, 7, 7], elementwise_affine=True,),
        )
        self.resnet_normalizer.apply(simple_conv_and_linear_weights_init)

        assert self.rgb_uuid == self.unshuffled_rgb_uuid

    def _create_visual_encoder(self) -> Optional[nn.Module]:
        """Create the visual encoder for the model."""
        return None

    @property
    def visual_encoder(self):
        # We make this a property as we don't want to register
        # self.map.resnet_l5 as a submodule of this module, doing
        # so would potentially overwriting the point of setting
        # `freeze_resnet_batchnorm` to `True` in the `ActiveNeuralSLAM`.
        return self.map.resnet_l5

    @visual_encoder.setter
    def visual_encoder(self, val: None):
        assert val is None, "Setting the visual encoder is not allowed."

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
        img = observations[self.rgb_uuid]
        nsteps, nsamplers, _, _, _ = img.shape

        img = img.permute(0, 1, 4, 2, 3)

        resnet_encoding = self.resnet_normalizer(
            self.visual_encoder(img.view(nsteps * nsamplers, *img.shape[-3:]))
        )

        x, rnn_hidden_states = self.state_encoder(
            self.resnet_features_downsampler(resnet_encoding.detach().clone()).view(
                nsteps, nsamplers, 512
            ),
            memory.tensor("rnn"),
            masks,
        )

        ac_output = ActorCriticOutput(
            distributions=self.actor(x), values=self.critic(x), extras={}
        )

        ego_map_logits = self.map.image_to_egocentric_map_logits(
            images=None, resnet_image_features=resnet_encoding
        )
        ego_map_logits = ego_map_logits.view(
            nsteps, nsamplers, *ego_map_logits.shape[-3:]
        )

        ac_output.extras["ego_height_binned_map_logits"] = ego_map_logits[
            :, :, : self.height_map_channels
        ].view(nsteps, nsamplers, -1, *ego_map_logits.shape[-2:])
        ac_output.extras["ego_semantic_map_logits"] = ego_map_logits[
            :, :, self.height_map_channels :
        ].view(nsteps, nsamplers, -1, *ego_map_logits.shape[-2:])
        return ac_output, memory.set_tensor("rnn", rnn_hidden_states)


class OnePhaseRearrangeActorCriticFrozenMap(ActorCriticModel[CategoricalDistr]):
    """A (IMG, MAP)->CNN->RNN actor-critic model for rearrangement tasks."""

    def __init__(
        self,
        map: ActiveNeuralSLAM,
        height_map_channels: int,
        semantic_map_channels: int,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        unshuffled_rgb_uuid: str,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid
        self.unshuffled_rgb_uuid = unshuffled_rgb_uuid

        self.concat_rgb_uuid = "concat_rgb"
        assert self.concat_rgb_uuid not in observation_space

        self.height_map_channels = height_map_channels
        self.semantic_map_channels = semantic_map_channels

        self.ego_map_encoder_out_dim = 512
        self.ego_map_attention = nn.Sequential(
            nn.Conv2d(
                3 * (height_map_channels + semantic_map_channels), 128, 2, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )
        self.ego_map_encoder = nn.Sequential(
            nn.Conv2d(
                3 * (height_map_channels + semantic_map_channels),
                self.ego_map_encoder_out_dim,
                2,
                stride=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ego_map_encoder_out_dim, self.ego_map_encoder_out_dim, 1,),
            nn.ReLU(inplace=True),
        )
        self.ego_map_attention.apply(simple_conv_and_linear_weights_init)
        self.ego_map_encoder.apply(simple_conv_and_linear_weights_init)

        #
        self.visual_attention = nn.Sequential(
            nn.Conv2d(3 * 512, 32, 1,), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 1,),
        )
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3 * 512, self._hidden_size, 1,), nn.ReLU(inplace=True),
        )
        self.visual_attention.apply(simple_conv_and_linear_weights_init)
        self.visual_encoder.apply(simple_conv_and_linear_weights_init)

        # Used to predict whether or not there is an object with a different pose
        # in front of the agent.
        self.sem_difference_predictor = nn.Linear(
            self.ego_map_encoder_out_dim, semantic_map_channels
        )
        self.sem_difference_predictor.apply(simple_conv_and_linear_weights_init)

        # Standard CNN
        self.state_encoder = RNNStateEncoder(
            self.ego_map_encoder_out_dim + self.recurrent_hidden_state_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )

        self.actor = LinearActorHead(self._hidden_size, action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

        self.map = map
        assert self.map.use_resnet_layernorm
        assert self.map.freeze_resnet_batchnorm
        for p in self.map.parameters():
            p.requires_grad = False

        self.train()

    def train(self, mode: bool = True):
        super(OnePhaseRearrangeActorCriticFrozenMap, self).train()
        self.map.eval()

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
            ),
            walkthrough_allo_map_probs=(
                (
                    ("sampler", None),
                    ("channels", self.map.n_map_channels),
                    ("height", self.map.map_size),
                    ("width", self.map.map_size),
                ),
                torch.float32,
            ),
        )

    def compute_visual_features(
        self, imgs: torch.Tensor,
    ):
        nsteps, nsamplers, h, w, c = imgs.shape

        return self.map.resnet_normalizer(
            self.map.resnet_l5(
                imgs.permute(0, 1, 4, 2, 3).reshape(nsteps * nsamplers, c, h, w)
            )
        ).view(nsteps, nsamplers, 512, 7, 7)

    def _create_visual_encoder(self) -> Optional[nn.Module]:
        """Create the visual encoder for the model."""
        return None

    def _get_height_binned_map_and_semantic_map(
        self, map: torch.Tensor, batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height_binned_map = map[:, :, : self.height_map_channels].view(
            batch_size, -1, *map.shape[-2:]
        )
        semantic_map = map[:, :, self.height_map_channels :].view(
            batch_size, -1, *map.shape[-2:]
        )
        return height_binned_map, semantic_map

    def forward(  # type:ignore
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        unshuffle_img = observations[self.rgb_uuid]
        walkthrough_img = observations[self.unshuffled_rgb_uuid]

        nsteps, nsamplers, h, w, c = unshuffle_img.shape

        with torch.no_grad():
            unshuffle_img_embed = self.compute_visual_features(unshuffle_img)
            walkthrough_img_embed = self.compute_visual_features(walkthrough_img)

        concat_img = torch.cat(
            (
                unshuffle_img_embed,
                walkthrough_img_embed,
                unshuffle_img_embed * walkthrough_img_embed,
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
        downsampled_img_embed = (
            (self.visual_encoder(concat_img_reshaped) * attention_probs)
            .mean(-1)
            .mean(-1)
        )
        downsampled_img_embed = downsampled_img_embed.view(*batch_shape, -1)

        ########
        dx_dz_drs_egocentric = observations["rel_position_change"]["dx_dz_dr"].clone()

        last_allo_pos = observations["rel_position_change"][
            "last_allocentric_position"
        ].clone()

        scene_bounds = observations["scene_bounds"]

        x_mins = scene_bounds["x_range"][..., 0]
        z_mins = scene_bounds["z_range"][..., 0]
        last_allo_pos_rel_bounds = last_allo_pos - torch.stack(
            (x_mins, z_mins, torch.zeros_like(x_mins)), dim=-1
        )

        # Converting THOR rotation to rotation expected by map
        last_allo_pos_rel_bounds[..., 2] = -last_allo_pos_rel_bounds[..., 2]
        dx_dz_drs_egocentric[..., 2] *= -1

        map_mask = masks.view(*masks.shape[:2], 1, 1, 1)

        walkthrough_allo_map_probs = memory.tensor("walkthrough_allo_map_probs")
        map_summaries = []
        rnn_hidden_states = memory.tensor("rnn")
        rnn_outputs_list = []

        for step in range(nsteps):
            with torch.no_grad():
                walkthrough_allo_map_probs = (  # Reset the map
                    walkthrough_allo_map_probs * map_mask[step]
                )
                walkthrough_map_result = self.map.forward(
                    images=None,
                    resnet_image_features=walkthrough_img_embed[step],
                    last_map_probs_allocentric=walkthrough_allo_map_probs,
                    last_xzrs_allocentric=last_allo_pos_rel_bounds[step].view(-1, 3),
                    dx_dz_drs_egocentric=dx_dz_drs_egocentric[step],
                    last_map_logits_egocentric=None,
                    return_allocentric_maps=True,
                )
                walkthrough_allo_map_probs = walkthrough_map_result[
                    "map_probs_allocentric_no_grad"
                ]

                unshuffle_map_result = self.map.forward(
                    images=None,
                    resnet_image_features=unshuffle_img_embed[step],
                    last_map_probs_allocentric=None,
                    last_xzrs_allocentric=last_allo_pos_rel_bounds[step].view(-1, 3),
                    dx_dz_drs_egocentric=dx_dz_drs_egocentric[step],
                    last_map_logits_egocentric=None,
                    return_allocentric_maps=False,
                )
                last_unshuffle_ego_map_logits = unshuffle_map_result[
                    "egocentric_update"
                ]

                walkthrough_updated_allo_probs = torch.sigmoid(
                    walkthrough_allo_map_probs
                )
                walkthrough_updated_ego_probs = self.map.allocentric_map_to_egocentric_view(
                    allocentric_map=walkthrough_updated_allo_probs,
                    xzr=walkthrough_map_result["xzr_allocentric_preds"],
                    padding_mode="zeros",
                )

                a = walkthrough_updated_ego_probs
                b = torch.sigmoid(last_unshuffle_ego_map_logits)

            concat_map = torch.cat((a, b, a * b,), dim=1,)
            attention_logits = self.ego_map_attention(concat_map)
            attention_probs = torch.softmax(
                attention_logits.view(concat_map.shape[0], -1), dim=-1,
            ).view(attention_logits.shape[0], 1, *attention_logits.shape[-2:])
            map_summary = (
                (self.ego_map_encoder(concat_map) * attention_probs).mean(-1).mean(-1)
            )
            map_summary = map_summary.view(concat_map.shape[0], -1)

            map_summaries.append(map_summary)

            x = torch.cat(
                (downsampled_img_embed[step], map_summary,), dim=-1,
            ).unsqueeze(0)
            x, rnn_hidden_states = self.state_encoder(
                x, rnn_hidden_states, masks[step : (step + 1)]
            )
            rnn_outputs_list.append(x)

        memory = memory.set_tensor(
            key="walkthrough_allo_map_probs", tensor=walkthrough_allo_map_probs
        )
        memory = memory.set_tensor(key="rnn", tensor=rnn_hidden_states)

        x = torch.cat(rnn_outputs_list, dim=0)
        extras = {}
        if torch.is_grad_enabled():
            # TODO: Create a loss to train the below as additonal supervision
            extras["object_type_change_logits"] = self.sem_difference_predictor(
                torch.stack(map_summaries, dim=0)
            )

        return (
            ActorCriticOutput(
                distributions=self.actor(x), values=self.critic(x), extras=extras,
            ),
            memory,
        )


class TwoPhaseRearrangeActorCriticFrozenMap(ActorCriticModel[CategoricalDistr]):
    """A (IMG, MAP)->CNN->RNN actor-critic model for rearrangement tasks."""

    def __init__(
        self,
        map: ActiveNeuralSLAM,
        height_map_channels: int,
        semantic_map_channels: int,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Dict,
        rgb_uuid: str,
        in_walkthrough_phase_uuid: str,
        is_walkthrough_phase_embedding_dim: int,
        done_action_index: int,
        walkthrougher_should_ignore_action_mask: Optional[Sequence[float]] = None,
        hidden_size=512,
        num_rnn_layers=1,
        rnn_type="GRU",
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self._hidden_size = hidden_size

        self.rgb_uuid = rgb_uuid
        self.in_walkthrough_phase_uuid = in_walkthrough_phase_uuid

        self.done_action_index = done_action_index

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

        self.height_map_channels = height_map_channels
        self.semantic_map_channels = semantic_map_channels

        self.ego_map_encoder_out_dim = 512
        self.ego_map_attention = nn.Sequential(
            nn.Conv2d(
                3 * (height_map_channels + semantic_map_channels), 128, 2, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )
        self.ego_map_encoder = nn.Sequential(
            nn.Conv2d(
                3 * (height_map_channels + semantic_map_channels),
                self.ego_map_encoder_out_dim,
                2,
                stride=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.ego_map_encoder_out_dim, self.ego_map_encoder_out_dim, 1,),
            nn.ReLU(inplace=True),
        )
        self.ego_map_attention.apply(simple_conv_and_linear_weights_init)
        self.ego_map_encoder.apply(simple_conv_and_linear_weights_init)

        #
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(512, 512, 1,),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((7, 7)),
            nn.Flatten(),
        )
        self.visual_encoder.apply(simple_conv_and_linear_weights_init)

        # Used to predict whether or not there is an object with a different pose
        # in front of the agent.
        self.sem_difference_predictor = nn.Linear(
            self.ego_map_encoder_out_dim, semantic_map_channels
        )
        self.sem_difference_predictor.apply(simple_conv_and_linear_weights_init)

        # Standard CNN
        self.state_encoder = RNNStateEncoder(
            self.ego_map_encoder_out_dim
            + is_walkthrough_phase_embedding_dim
            + 2 * self.recurrent_hidden_state_size,
            self._hidden_size,
            num_layers=num_rnn_layers,
            rnn_type=rnn_type,
        )
        self.state_encoder.apply(simple_conv_and_linear_weights_init)

        self.walkthrough_encoder = RNNStateEncoder(
            self._hidden_size, self._hidden_size, num_layers=1, rnn_type="GRU",
        )
        self.walkthrough_encoder.apply(simple_conv_and_linear_weights_init)

        self.walkthrough_ac = LinearActorCriticHead(self._hidden_size, action_space.n)
        self.walkthrough_ac.actor_and_critic.bias.data[self.done_action_index] -= 3
        self.unshuffle_ac = LinearActorCriticHead(self._hidden_size, action_space.n)

        self.map = map
        assert self.map.use_resnet_layernorm
        assert self.map.freeze_resnet_batchnorm
        for p in self.map.parameters():
            p.requires_grad = False

        self.train()

    def train(self, mode: bool = True):
        super(TwoPhaseRearrangeActorCriticFrozenMap, self).train()
        self.map.eval()

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
            ),
            walkthrough_encoding=(
                (
                    ("layer", self.walkthrough_encoder.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            ),
            walkthrough_allo_map_probs=(
                (
                    ("sampler", None),
                    ("channels", self.map.n_map_channels),
                    ("height", self.map.map_size),
                    ("width", self.map.map_size),
                ),
                torch.float32,
            ),
        )

    def compute_visual_features(
        self, imgs: torch.Tensor,
    ):
        nsteps, nsamplers, h, w, c = imgs.shape

        return self.map.resnet_normalizer(
            self.map.resnet_l5(
                imgs.permute(0, 1, 4, 2, 3).reshape(nsteps * nsamplers, c, h, w)
            )
        ).view(nsteps, nsamplers, 512, 7, 7)

    def _create_visual_encoder(self) -> Optional[nn.Module]:
        """Create the visual encoder for the model."""
        return None

    def _get_height_binned_map_and_semantic_map(
        self, map: torch.Tensor, batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height_binned_map = map[:, :, : self.height_map_channels].view(
            batch_size, -1, *map.shape[-2:]
        )
        semantic_map = map[:, :, self.height_map_channels :].view(
            batch_size, -1, *map.shape[-2:]
        )
        return height_binned_map, semantic_map

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
        masks_with_unshuffle_reset = masks.float()
        del masks  # Just to make sure we don't accidentally use `masks when we want `masks_no_unshuffle_reset`

        cur_img = observations[self.rgb_uuid]

        nsteps, nsamplers, h, w, c = cur_img.shape

        with torch.no_grad():
            cur_img_embed = self.compute_visual_features(cur_img)

        batch_shape, features_shape = cur_img_embed.shape[:-3], cur_img_embed.shape[-3:]
        concat_img_reshaped = cur_img_embed.view(-1, *features_shape)
        downsampled_img_embed = self.visual_encoder(concat_img_reshaped).view(
            *batch_shape, -1
        )

        # Various embeddings
        is_walkthrough_phase_embedding = self.is_walkthrough_phase_embedder(
            in_walkthrough_phase_mask.long()
        ).squeeze(-2)

        #######
        dx_dz_drs_egocentric = observations["rel_position_change"]["dx_dz_dr"].clone()

        last_allo_pos = observations["rel_position_change"][
            "last_allocentric_position"
        ].clone()

        scene_bounds = observations["scene_bounds"]

        x_mins = scene_bounds["x_range"][..., 0]
        z_mins = scene_bounds["z_range"][..., 0]
        last_allo_pos_rel_bounds = last_allo_pos - torch.stack(
            (x_mins, z_mins, torch.zeros_like(x_mins)), dim=-1
        )

        # Converting THOR rotation to rotation expected by map
        last_allo_pos_rel_bounds[..., 2] = -last_allo_pos_rel_bounds[..., 2]
        dx_dz_drs_egocentric[..., 2] *= -1

        map_mask = masks_no_unshuffle_reset.view(nsteps, nsamplers, 1, 1, 1)
        in_walkthrough_map_mask = in_walkthrough_float.view(nsteps, nsamplers, 1, 1, 1)
        in_unshuffle_map_mask = in_unshuffle_float.view(nsteps, nsamplers, 1, 1, 1)

        walkthrough_allo_map_probs = memory.tensor("walkthrough_allo_map_probs")
        walkthrough_encoding = memory.tensor("walkthrough_encoding")
        map_summaries = []
        rnn_hidden_states = memory.tensor("rnn")
        rnn_outputs_list = []
        for step in range(nsteps):
            with torch.no_grad():
                walkthrough_allo_map_probs = (  # Resetting the map
                    walkthrough_allo_map_probs * map_mask[step]
                )
                map_result = self.map.forward(
                    images=None,
                    resnet_image_features=cur_img_embed[step],
                    last_map_probs_allocentric=walkthrough_allo_map_probs,
                    last_xzrs_allocentric=last_allo_pos_rel_bounds[step].view(-1, 3),
                    dx_dz_drs_egocentric=dx_dz_drs_egocentric[step],
                    last_map_logits_egocentric=None,
                    return_allocentric_maps=True,
                )
                walkthrough_allo_map_probs = (
                    map_result["map_probs_allocentric_no_grad"]
                    * in_walkthrough_map_mask[step]
                    + walkthrough_allo_map_probs * in_unshuffle_map_mask[step]
                )

                walkthrough_updated_ego_probs = self.map.allocentric_map_to_egocentric_view(
                    allocentric_map=walkthrough_allo_map_probs,
                    xzr=map_result["xzr_allocentric_preds"],
                    padding_mode="zeros",
                )

                last_map_logits_egocentric = map_result["egocentric_update"]
                a = walkthrough_updated_ego_probs
                b = torch.sigmoid(last_map_logits_egocentric)

            concat_map = torch.cat((a, b, a * b,), dim=1,)
            attention_logits = self.ego_map_attention(concat_map)
            attention_probs = torch.softmax(
                attention_logits.view(concat_map.shape[0], -1), dim=-1,
            ).view(attention_logits.shape[0], 1, *attention_logits.shape[-2:])
            map_summary = (
                (self.ego_map_encoder(concat_map) * attention_probs).mean(-1).mean(-1)
            )
            map_summary = map_summary.view(concat_map.shape[0], -1)

            map_summaries.append(map_summary)

            rnn_input = torch.cat(
                (
                    downsampled_img_embed[step],
                    map_summary,
                    walkthrough_encoding[0] * masks_no_unshuffle_reset[step],
                    is_walkthrough_phase_embedding[step],
                ),
                dim=-1,
            ).unsqueeze(0)
            rnn_out, rnn_hidden_states = self.state_encoder(
                rnn_input,
                rnn_hidden_states,
                masks_with_unshuffle_reset[step : (step + 1)],
            )
            rnn_outputs_list.append(rnn_out)

            new_walkthrough_encoding, _ = self.walkthrough_encoder(
                rnn_out,
                walkthrough_encoding,
                masks_no_unshuffle_reset[step : step + 1],
            )
            walkthrough_encoding = (
                walkthrough_encoding * in_unshuffle_float[step : step + 1]
                + new_walkthrough_encoding * in_walkthrough_float[step : step + 1]
            )

        memory = memory.set_tensor("walkthrough_encoding", walkthrough_encoding)
        memory = memory.set_tensor(
            key="walkthrough_allo_map_probs", tensor=walkthrough_allo_map_probs
        )
        memory = memory.set_tensor(key="rnn", tensor=rnn_hidden_states)

        rnn_out = torch.cat(rnn_outputs_list, dim=0)
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

        return ActorCriticOutput(distributions=actor, values=values, extras={}), memory
