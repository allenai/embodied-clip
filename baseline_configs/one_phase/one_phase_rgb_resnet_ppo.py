from baseline_configs.one_phase.one_phase_rgb_ppo import OnePhaseRGBPPOExperimentConfig


class OnePhaseRGBResNetPPOExperimentConfig(OnePhaseRGBPPOExperimentConfig):
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN18", "imagenet")

    @classmethod
    def tag(cls) -> str:
        return "OnePhaseRGBResNetPPO"
