from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import ObjectNavRoboThorBaseConfig

class ZeroshotObjectNavRoboThorBaseConfig(ObjectNavRoboThorBaseConfig):

    # SEEN_TARGET_TYPES = tuple(
    #     sorted(
    #         [
    #             "AlarmClock",
    #             "BaseballBat",
    #             "Bowl",
    #             "GarbageCan",
    #             "Laptop",
    #             "Mug",
    #             "SprayBottle",
    #             "Vase",
    #         ]
    #     )
    # )

    # UNSEEN_TARGET_TYPES = tuple(
    #     sorted(
    #         [
    #             "Apple",
    #             "BasketBall",
    #             "HousePlant",
    #             "Television"
    #         ]
    #     )
    # )

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
