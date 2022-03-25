from baseline_configs.two_phase.two_phase_rgb_ppowalkthrough_ilunshuffle import (
    TwoPhaseRGBPPOWalkthroughILUnshuffleExperimentConfig,
)


class TwoPhaseRGBResNetPPOWalkthroughILUnshuffleExperimentConfig(
    TwoPhaseRGBPPOWalkthroughILUnshuffleExperimentConfig
):
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN18", "imagenet")
    IL_PIPELINE_TYPE: str = "40proc-longtf"

    @classmethod
    def tag(cls) -> str:
        return f"TwoPhaseRGBResNetPPOWalkthroughILUnshuffle_{cls.IL_PIPELINE_TYPE}"
