from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import ObjectNavRoboThorBaseConfig

class ZeroshotObjectNavRoboThorBaseConfig(ObjectNavRoboThorBaseConfig):

    TARGET_TYPES = tuple(
        sorted(
            [
                "AlarmClock",
                "BaseballBat",
                "Bowl",
                "GarbageCan",
                "Laptop",
                "Mug",
                "SprayBottle",
                "Vase",
            ]
        )
    )
