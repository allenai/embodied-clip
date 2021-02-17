from baseline_configs.walkthrough.walkthrough_rgb_ppo import (
    WalkthroughPPOExperimentConfig,
)


class WalkthroughRGBResNetPPOExperimentConfig(WalkthroughPPOExperimentConfig):
    USE_RESNET_CNN = True

    @classmethod
    def tag(cls) -> str:
        return "WalkthroughRGBResNetPPO"
