from baseline_configs.one_phase.one_phase_rgb_il_base import (
    OnePhaseRGBILBaseExperimentConfig,
)


class OnePhaseRGBResNetDaggerExperimentConfig(OnePhaseRGBILBaseExperimentConfig):
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN50", "imagenet")
    IL_PIPELINE_TYPE = "40proc"

    @classmethod
    def tag(cls) -> str:
        return f"OnePhaseRGBResNetDagger_{cls.IL_PIPELINE_TYPE}"
