from typing import Dict, Any

from allenact.algorithms.onpolicy_sync.losses import PPO
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.utils.experiment_utils import LinearDecay, PipelineStage

from baseline_configs.one_phase.one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)


class OnePhasePPORGBExperimentConfig(OnePhaseRGBBaseExperimentConfig):
    USE_RESNET_CNN = False

    @classmethod
    def tag(cls) -> str:
        return "OnePhaseRGBPPO"

    @classmethod
    def num_train_processes(cls) -> int:
        return 40

    @classmethod
    def _training_pipeline_info(cls, **kwargs) -> Dict[str, Any]:
        """Define how the model trains."""

        training_steps = cls.TRAINING_STEPS
        return dict(
            named_losses=dict(
                ppo_loss=PPO(clip_decay=LinearDecay(training_steps), **PPOConfig)
            ),
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=training_steps,)
            ],
            num_steps=64,
            num_mini_batch=1,
            update_repeats=3,
            use_lr_decay=True,
            lr=3e-4,
        )
