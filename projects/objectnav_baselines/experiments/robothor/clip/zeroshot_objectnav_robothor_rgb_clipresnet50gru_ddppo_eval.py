from projects.objectnav_baselines.experiments.robothor.clip.zeroshot_objectnav_robothor_rgb_clipresnet50gru_ddppo import ZeroshotObjectNavRoboThorClipRGBPPOExperimentConfig


class EvalZeroshotObjectNavRoboThorClipRGBPPOExperimentConfig(ZeroshotObjectNavRoboThorClipRGBPPOExperimentConfig):

    TARGET_TYPES = tuple(
        sorted(
            [
                "AlarmClock",
                "Apple",
                "BaseballBat",
                "BasketBall",
                "Bowl",
                "GarbageCan",
                "HousePlant",
                "Laptop",
                "Mug",
                "SprayBottle",
                "Television",
                "Vase",
            ]
        )
    )
