from baseline_configs.one_phase.one_phase_rgb_ppo import OnePhaseRGBPPOExperimentConfig


class OnePhaseRGBResNetPPOExperimentConfig(OnePhaseRGBPPOExperimentConfig):
    USE_RESNET_CNN = True

    @classmethod
    def tag(cls) -> str:
        return "OnePhaseRGBResNetPPO"
