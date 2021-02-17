from typing import Dict, Any

from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.algorithms.onpolicy_sync.losses.ppo import PPOConfig
from allenact.base_abstractions.sensor import ExpertActionSensor
from allenact.utils.experiment_utils import LinearDecay, PipelineStage

from baseline_configs.one_phase.one_phase_rgb_il_base import (
    il_training_params,
    StepwiseLinearDecay,
)
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from baseline_configs.two_phase.two_phase_rgb_base import (
    TwoPhaseRGBBaseExperimentConfig,
)
from rearrange.losses import MaskedPPO


class TwoPhaseRGBPPOWalkthroughILUnshuffleExperimentConfig(
    TwoPhaseRGBBaseExperimentConfig
):
    SENSORS = [
        *TwoPhaseRGBBaseExperimentConfig.SENSORS,
        ExpertActionSensor(len(RearrangeBaseExperimentConfig.actions())),
    ]

    USE_RESNET_CNN = False
    IL_PIPELINE_TYPE: str = "40proc-longtf"

    @classmethod
    def tag(cls) -> str:
        return f"TwoPhaseRGBPPOWalkthroughILUnshuffle_{cls.IL_PIPELINE_TYPE}"

    @classmethod
    def num_train_processes(cls) -> int:
        return cls._use_label_to_get_training_params()["num_train_processes"]

    @classmethod
    def _training_pipeline_info(cls) -> Dict[str, Any]:
        """Define how the model trains."""

        training_steps = cls.TRAINING_STEPS
        il_params = cls._use_label_to_get_training_params()
        bc_tf1_steps = il_params["bc_tf1_steps"]
        dagger_steps = il_params["dagger_steps"]

        return dict(
            named_losses=dict(
                walkthrough_ppo_loss=MaskedPPO(
                    mask_uuid="in_walkthrough_phase",
                    ppo_params=dict(
                        clip_decay=LinearDecay(training_steps), **PPOConfig
                    ),
                ),
                imitation_loss=Imitation(),
            ),
            pipeline_stages=[
                PipelineStage(
                    loss_names=["walkthrough_ppo_loss", "imitation_loss"],
                    max_stage_steps=training_steps,
                    teacher_forcing=StepwiseLinearDecay(
                        cumm_steps_and_values=[
                            (bc_tf1_steps, 1.0),
                            (bc_tf1_steps + dagger_steps, 0.0),
                        ]
                    ),
                )
            ],
            **il_params,
        )

    @classmethod
    def _use_label_to_get_training_params(cls):
        return il_training_params(
            label=cls.IL_PIPELINE_TYPE.lower(), training_steps=cls.TRAINING_STEPS
        )
