from baseline_configs.one_phase.one_phase_rgb_ppo import OnePhasePPORGBExperimentConfig


class OnePhasePPORGBResNetExperimentConfig(OnePhasePPORGBExperimentConfig):
    USE_RESNET_CNN = True

    @classmethod
    def tag(cls) -> str:
        return "OnePhaseRGBResNetPPO"
