from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)


class OnePhaseRGBCompassResNetDaggerExperimentConfig(OnePhaseRGBILBaseExperimentConfig):
    USE_RESNET_CNN = True
    IL_PIPELINE_TYPE = "40proc"

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBResNetDagger_{cls.IL_PIPELINE_TYPE}"
