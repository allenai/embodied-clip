import os
from pathlib import Path

ABS_PATH_OF_REARRANGE_TOP_LEVEL_DIR = os.path.abspath(os.path.dirname(Path(__file__)))
IOU_THRESHOLD = 0.5
OPENNESS_THRESHOLD = 0.2
POSITION_DIFF_BARRIER = 2.0
