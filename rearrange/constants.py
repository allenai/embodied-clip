import os
from pathlib import Path

MAX_HAND_METERS = 0.5
FOV = 90

REQUIRED_THOR_VERSION = ">=2.7.2"
STARTER_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(Path(__file__))), "../data"
)
THOR_COMMIT_ID = "5b20c5692d51c6f3c3596803c491c3da8d43eb2c"
STEP_SIZE = 0.25

# fmt: off
REARRANGE_SIM_OBJECTS = [
    # A
    "AlarmClock", "AluminumFoil", "Apple", "AppleSliced", "ArmChair",
    "BaseballBat", "BasketBall", "Bathtub", "BathtubBasin", "Bed", "Blinds", "Book", "Boots", "Bottle", "Bowl", "Box",
    # B
    "Bread", "BreadSliced", "ButterKnife",
    # C
    "Cabinet", "Candle", "CD", "CellPhone", "Chair", "Cloth", "CoffeeMachine", "CoffeeTable", "CounterTop", "CreditCard",
    "Cup", "Curtains",
    # D
    "Desk", "DeskLamp", "Desktop", "DiningTable", "DishSponge", "DogBed", "Drawer", "Dresser", "Dumbbell",
    # E
    "Egg", "EggCracked",
    # F
    "Faucet", "Floor", "FloorLamp", "Footstool", "Fork", "Fridge",
    # G
    "GarbageBag", "GarbageCan",
    # H
    "HandTowel", "HandTowelHolder", "HousePlant", "Kettle", "KeyChain", "Knife",
    # L
    "Ladle", "Laptop", "LaundryHamper", "Lettuce", "LettuceSliced", "LightSwitch",
    # M
    "Microwave", "Mirror", "Mug",
    # N
    "Newspaper",
    # O
    "Ottoman",
    # P
    "Painting", "Pan", "PaperTowel", "Pen", "Pencil", "PepperShaker", "Pillow", "Plate", "Plunger", "Poster", "Pot",
    "Potato", "PotatoSliced",
    # R
    "RemoteControl", "RoomDecor",
    # S
    "Safe", "SaltShaker", "ScrubBrush", "Shelf", "ShelvingUnit", "ShowerCurtain", "ShowerDoor", "ShowerGlass",
    "ShowerHead", "SideTable", "Sink", "SinkBasin", "SoapBar", "SoapBottle", "Sofa", "Spatula", "Spoon", "SprayBottle",
    "Statue", "Stool", "StoveBurner", "StoveKnob",
    # T
    "TableTopDecor", "TargetCircle", "TeddyBear", "Television", "TennisRacket", "TissueBox", "Toaster", "Toilet",
    "ToiletPaper", "ToiletPaperHanger", "Tomato", "TomatoSliced", "Towel", "TowelHolder", "TVStand",
    # V
    "VacuumCleaner", "Vase",
    # W
    "Watch", "WateringCan", "Window", "WineBottle",
]
# fmt: on


# fmt: off
OBJECT_TYPES_WITH_PROPERTIES = {
    "StoveBurner": {"openable": False, "receptacle": True, "pickupable": False},
    "Drawer": {"openable": True, "receptacle": True, "pickupable": False},
    "CounterTop": {"openable": False, "receptacle": True, "pickupable": False},
    "Cabinet": {"openable": True, "receptacle": True, "pickupable": False},
    "StoveKnob": {"openable": False, "receptacle": False, "pickupable": False},
    "Window": {"openable": False, "receptacle": False, "pickupable": False},
    "Sink": {"openable": False, "receptacle": True, "pickupable": False},
    "Floor": {"openable": False, "receptacle": True, "pickupable": False},
    "Book": {"openable": True, "receptacle": False, "pickupable": True},
    "Bottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Knife": {"openable": False, "receptacle": False, "pickupable": True},
    "Microwave": {"openable": True, "receptacle": True, "pickupable": False},
    "Bread": {"openable": False, "receptacle": False, "pickupable": True},
    "Fork": {"openable": False, "receptacle": False, "pickupable": True},
    "Shelf": {"openable": False, "receptacle": True, "pickupable": False},
    "Potato": {"openable": False, "receptacle": False, "pickupable": True},
    "HousePlant": {"openable": False, "receptacle": False, "pickupable": False},
    "Toaster": {"openable": False, "receptacle": True, "pickupable": False},
    "SoapBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Kettle": {"openable": True, "receptacle": False, "pickupable": True},
    "Pan": {"openable": False, "receptacle": True, "pickupable": True},
    "Plate": {"openable": False, "receptacle": True, "pickupable": True},
    "Tomato": {"openable": False, "receptacle": False, "pickupable": True},
    "Vase": {"openable": False, "receptacle": False, "pickupable": True},
    "GarbageCan": {"openable": False, "receptacle": True, "pickupable": False},
    "Egg": {"openable": False, "receptacle": False, "pickupable": True},
    "CreditCard": {"openable": False, "receptacle": False, "pickupable": True},
    "WineBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Pot": {"openable": False, "receptacle": True, "pickupable": True},
    "Spatula": {"openable": False, "receptacle": False, "pickupable": True},
    "PaperTowelRoll": {"openable": False, "receptacle": False, "pickupable": True},
    "Cup": {"openable": False, "receptacle": True, "pickupable": True},
    "Fridge": {"openable": True, "receptacle": True, "pickupable": False},
    "CoffeeMachine": {"openable": False, "receptacle": True, "pickupable": False},
    "Bowl": {"openable": False, "receptacle": True, "pickupable": True},
    "SinkBasin": {"openable": False, "receptacle": True, "pickupable": False},
    "SaltShaker": {"openable": False, "receptacle": False, "pickupable": True},
    "PepperShaker": {"openable": False, "receptacle": False, "pickupable": True},
    "Lettuce": {"openable": False, "receptacle": False, "pickupable": True},
    "ButterKnife": {"openable": False, "receptacle": False, "pickupable": True},
    "Apple": {"openable": False, "receptacle": False, "pickupable": True},
    "DishSponge": {"openable": False, "receptacle": False, "pickupable": True},
    "Spoon": {"openable": False, "receptacle": False, "pickupable": True},
    "LightSwitch": {"openable": False, "receptacle": False, "pickupable": False},
    "Mug": {"openable": False, "receptacle": True, "pickupable": True},
    "ShelvingUnit": {"openable": False, "receptacle": True, "pickupable": False},
    "Statue": {"openable": False, "receptacle": False, "pickupable": True},
    "Stool": {"openable": False, "receptacle": True, "pickupable": False},
    "Faucet": {"openable": False, "receptacle": False, "pickupable": False},
    "Ladle": {"openable": False, "receptacle": False, "pickupable": True},
    "CellPhone": {"openable": False, "receptacle": False, "pickupable": True},
    "Chair": {"openable": False, "receptacle": True, "pickupable": False},
    "SideTable": {"openable": False, "receptacle": True, "pickupable": False},
    "DiningTable": {"openable": False, "receptacle": True, "pickupable": False},
    "Pen": {"openable": False, "receptacle": False, "pickupable": True},
    "SprayBottle": {"openable": False, "receptacle": False, "pickupable": True},
    "Curtains": {"openable": False, "receptacle": False, "pickupable": False},
    "Pencil": {"openable": False, "receptacle": False, "pickupable": True},
    "Blinds": {"openable": True, "receptacle": False, "pickupable": False},
    "GarbageBag": {"openable": False, "receptacle": False, "pickupable": False},
    "Safe": {"openable": True, "receptacle": True, "pickupable": False},
    "Painting": {"openable": False, "receptacle": False, "pickupable": False},
    "Box": {"openable": True, "receptacle": True, "pickupable": True},
    "Laptop": {"openable": True, "receptacle": False, "pickupable": True},
    "Television": {"openable": False, "receptacle": False, "pickupable": False},
    "TissueBox": {"openable": False, "receptacle": False, "pickupable": True},
    "KeyChain": {"openable": False, "receptacle": False, "pickupable": True},
    "FloorLamp": {"openable": False, "receptacle": False, "pickupable": False},
    "DeskLamp": {"openable": False, "receptacle": False, "pickupable": False},
    "Pillow": {"openable": False, "receptacle": False, "pickupable": True},
    "RemoteControl": {"openable": False, "receptacle": False, "pickupable": True},
    "Watch": {"openable": False, "receptacle": False, "pickupable": True},
    "Newspaper": {"openable": False, "receptacle": False, "pickupable": True},
    "ArmChair": {"openable": False, "receptacle": True, "pickupable": False},
    "CoffeeTable": {"openable": False, "receptacle": True, "pickupable": False},
    "TVStand": {"openable": False, "receptacle": True, "pickupable": False},
    "Sofa": {"openable": False, "receptacle": True, "pickupable": False},
    "WateringCan": {"openable": False, "receptacle": False, "pickupable": True},
    "Boots": {"openable": False, "receptacle": False, "pickupable": True},
    "Ottoman": {"openable": False, "receptacle": True, "pickupable": False},
    "Desk": {"openable": False, "receptacle": True, "pickupable": False},
    "Dresser": {"openable": False, "receptacle": True, "pickupable": False},
    "Mirror": {"openable": False, "receptacle": False, "pickupable": False},
    "DogBed": {"openable": False, "receptacle": True, "pickupable": False},
    "Candle": {"openable": False, "receptacle": False, "pickupable": True},
    "RoomDecor": {"openable": False, "receptacle": False, "pickupable": False},
    "Bed": {"openable": False, "receptacle": True, "pickupable": False},
    "BaseballBat": {"openable": False, "receptacle": False, "pickupable": True},
    "BasketBall": {"openable": False, "receptacle": False, "pickupable": True},
    "AlarmClock": {"openable": False, "receptacle": False, "pickupable": True},
    "CD": {"openable": False, "receptacle": False, "pickupable": True},
    "TennisRacket": {"openable": False, "receptacle": False, "pickupable": True},
    "TeddyBear": {"openable": False, "receptacle": False, "pickupable": True},
    "Poster": {"openable": False, "receptacle": False, "pickupable": False},
    "Cloth": {"openable": False, "receptacle": False, "pickupable": True},
    "Dumbbell": {"openable": False, "receptacle": False, "pickupable": True},
    "LaundryHamper": {"openable": True, "receptacle": True, "pickupable": False},
    "TableTopDecor": {"openable": False, "receptacle": False, "pickupable": True},
    "Desktop": {"openable": False, "receptacle": False, "pickupable": False},
    "Footstool": {"openable": False, "receptacle": True, "pickupable": True},
    "BathtubBasin": {"openable": False, "receptacle": True, "pickupable": False},
    "ShowerCurtain": {"openable": True, "receptacle": False, "pickupable": False},
    "ShowerHead": {"openable": False, "receptacle": False, "pickupable": False},
    "Bathtub": {"openable": False, "receptacle": True, "pickupable": False},
    "Towel": {"openable": False, "receptacle": False, "pickupable": True},
    "HandTowel": {"openable": False, "receptacle": False, "pickupable": True},
    "Plunger": {"openable": False, "receptacle": False, "pickupable": True},
    "TowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
    "ToiletPaperHanger": {"openable": False, "receptacle": True, "pickupable": False},
    "SoapBar": {"openable": False, "receptacle": False, "pickupable": True},
    "ToiletPaper": {"openable": False, "receptacle": False, "pickupable": True},
    "HandTowelHolder": {"openable": False, "receptacle": True, "pickupable": False},
    "ScrubBrush": {"openable": False, "receptacle": False, "pickupable": True},
    "Toilet": {"openable": True, "receptacle": True, "pickupable": False},
    "ShowerGlass": {"openable": False, "receptacle": False, "pickupable": False},
    "ShowerDoor": {"openable": True, "receptacle": False, "pickupable": False},
    "AluminumFoil": {"openable": False, "receptacle": False, "pickupable": True},
    "VacuumCleaner": {"openable": False, "receptacle": False, "pickupable": False}
}
# fmt: on
