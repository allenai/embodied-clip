from baseline_configs.walkthrough.walkthrough_rgb_ppo import (
    WalkthroughPPOExperimentConfig,
)


class WalkthroughRGBResNetPPOExperimentConfig(WalkthroughPPOExperimentConfig):
    CNN_PREPROCESSOR_TYPE_AND_PRETRAINING = ("RN18", "imagenet")

    @classmethod
    def tag(cls) -> str:
        return "WalkthroughRGBResNetPPO"
