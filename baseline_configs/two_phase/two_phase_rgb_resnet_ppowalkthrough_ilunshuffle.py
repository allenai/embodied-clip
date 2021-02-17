from baseline_configs.two_phase.two_phase_rgb_ppowalkthrough_ilunshuffle import (
    TwoPhaseRGBPPOWalkthroughILUnshuffleExperimentConfig,
)


class TwoPhaseRGBResNetPPOWalkthroughILUnshuffleExperimentConfig(
    TwoPhaseRGBPPOWalkthroughILUnshuffleExperimentConfig
):
    USE_RESNET_CNN = True
    IL_PIPELINE_TYPE: str = "40proc-longtf"

    @classmethod
    def tag(cls) -> str:
        return f"TwoPhaseRGBResNetPPOWalkthroughILUnshuffle_{cls.IL_PIPELINE_TYPE}"
