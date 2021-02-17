from typing import Tuple, Sequence, Optional, Dict, Any

import torch
from allenact.algorithms.onpolicy_sync.losses.imitation import Imitation
from allenact.base_abstractions.sensor import ExpertActionSensor
from allenact.utils.experiment_utils import PipelineStage
from allenact.utils.misc_utils import all_unique

from baseline_configs.one_phase.one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)
from baseline_configs.rearrange_base import RearrangeBaseExperimentConfig


class StepwiseLinearDecay:
    def __init__(self, cumm_steps_and_values: Sequence[Tuple[int, float]]):
        assert len(cumm_steps_and_values) >= 1

        self.steps_and_values = list(sorted(cumm_steps_and_values))
        self.steps = [steps for steps, _ in cumm_steps_and_values]
        self.values = [value for _, value in cumm_steps_and_values]

        assert all_unique(self.steps)
        assert all(0 <= v <= 1 for v in self.values)

    def __call__(self, epoch: int) -> float:
        """Get the value for the input number of steps."""
        if epoch <= self.steps[0]:
            return self.values[0]
        elif epoch >= self.steps[-1]:
            return self.values[-1]
        else:
            # TODO: Binary search would be more efficient but seems overkill
            for i, (s0, s1) in enumerate(zip(self.steps[:-1], self.steps[1:])):
                if epoch < s1:
                    p = (epoch - s0) / (s1 - s0)
                    v0 = self.values[i]
                    v1 = self.values[i + 1]
                    return p * v1 + (1 - p) * v0


def il_training_params(label: str, training_steps: int):
    use_lr_decay = False

    if label == "80proc":
        lr = 3e-4
        num_train_processes = 80
        num_steps = 64
        dagger_steps = min(int(1e6), training_steps // 10)
        bc_tf1_steps = min(int(1e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 2 if torch.cuda.is_available() else 1

    elif label == "40proc":
        lr = 3e-4
        num_train_processes = 40
        num_steps = 64
        dagger_steps = min(int(1e6), training_steps // 10)
        bc_tf1_steps = min(int(1e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 1

    elif label == "40proc-longtf":
        lr = 3e-4
        num_train_processes = 40
        num_steps = 64
        dagger_steps = min(int(5e6), training_steps // 10)
        bc_tf1_steps = min(int(5e5), training_steps // 10)
        update_repeats = 3
        num_mini_batch = 1

    else:
        raise NotImplementedError

    return dict(
        lr=lr,
        num_steps=num_steps,
        num_mini_batch=num_mini_batch,
        update_repeats=update_repeats,
        use_lr_decay=use_lr_decay,
        num_train_processes=num_train_processes,
        dagger_steps=dagger_steps,
        bc_tf1_steps=bc_tf1_steps,
    )


class OnePhaseRGBILBaseExperimentConfig(OnePhaseRGBBaseExperimentConfig):
    SENSORS = [
        *OnePhaseRGBBaseExperimentConfig.SENSORS,
        ExpertActionSensor(len(RearrangeBaseExperimentConfig.actions())),
    ]

    IL_PIPELINE_TYPE: Optional[str] = None

    @classmethod
    def _training_pipeline_info(cls, **kwargs) -> Dict[str, Any]:
        """Define how the model trains."""

        training_steps = cls.TRAINING_STEPS
        params = cls._use_label_to_get_training_params()
        bc_tf1_steps = params["bc_tf1_steps"]
        dagger_steps = params["dagger_steps"]

        return dict(
            named_losses=dict(imitation_loss=Imitation()),
            pipeline_stages=[
                PipelineStage(
                    loss_names=["imitation_loss"],
                    max_stage_steps=training_steps,
                    teacher_forcing=StepwiseLinearDecay(
                        cumm_steps_and_values=[
                            (bc_tf1_steps, 1.0),
                            (bc_tf1_steps + dagger_steps, 0.0),
                        ]
                    ),
                )
            ],
            **params
        )

    @classmethod
    def num_train_processes(cls) -> int:
        return cls._use_label_to_get_training_params()["num_train_processes"]

    @classmethod
    def _use_label_to_get_training_params(cls):
        return il_training_params(
            label=cls.IL_PIPELINE_TYPE.lower(), training_steps=cls.TRAINING_STEPS
        )
