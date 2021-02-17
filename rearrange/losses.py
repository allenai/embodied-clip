from typing import cast, Dict, Any

import torch
from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
)
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput


class MaskedPPO(AbstractActorCriticLoss):
    """Compute the PPO loss where specified by a mask.

    # Attributes
    mask_uuid : A string specifying the sensor UUID to use for masking. The PPO loss will only
        be computed for those steps where this mask equals 1.
    """

    def __init__(
        self, mask_uuid: str, ppo_params: Dict[str, Any],
    ):
        """Initializer.

        # Parameters
        mask_uuid : A string specifying the sensor UUID to use for masking. The PPO loss will only
            be computed for those steps where this mask equals 1.
        ppo_params : A dictionary containing keyword arguments for the ppo loss. See the `PPO` class
            for what arguments are available.
        """
        super().__init__()
        self.mask_uuid = mask_uuid
        self._ppo_loss = PPO(**ppo_params)

    def loss(  # type: ignore
        self,
        step_count: int,
        batch: ObservationType,
        actor_critic_output: ActorCriticOutput[CategoricalDistr],
        *args,
        **kwargs
    ):
        mask = batch["observations"][self.mask_uuid].float()
        denominator = mask.sum().item()

        losses_per_step = self._ppo_loss.loss_per_step(
            step_count=step_count, batch=batch, actor_critic_output=actor_critic_output,
        )
        losses_per_step["entropy"] = (
            losses_per_step["entropy"][0].unsqueeze(-1),
            losses_per_step["entropy"][1],
        )
        losses = {
            key: ((loss * mask).sum() / max(denominator, 1), weight)
            for (key, (loss, weight)) in losses_per_step.items()
        }

        total_loss = sum(
            loss * weight if weight is not None else loss
            for loss, weight in losses.values()
        )

        if denominator == 0:
            losses_to_record = {}
        else:
            losses_to_record = {
                "ppo_total": cast(torch.Tensor, total_loss).item(),
                **{key: loss.item() for key, (loss, _) in losses.items()},
            }

        return (
            total_loss,
            losses_to_record,
        )
