from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)


class OnePhaseRGBDaggerExperimentConfig(OnePhaseRGBILBaseExperimentConfig):
    USE_RESNET_CNN = False
    IL_PIPELINE_TYPE = "40proc"

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBDagger_{cls.IL_PIPELINE_TYPE}"
