# thor-rearrange-starter

Install AI2-THOR with
```
pip install ai2thor==2.4.9
```
Then use the starting template in `main.py` to leverage a running example:
```python
from rearrange_config import Controller
controller = Controller(stage='train')
for i_episode in range(20):
    controller.reset()
    # walkthrough
    for t_step in range(500):
        rgb_observation = controller.last_event.frame
        controller.step('MoveAhead')  # or any other action
    controller.shuffle()
    # unshuffle
    for t_step in range(500):
        rgb_observation = controller.last_event.frame
        controller.step('MoveAhead')  # or any other action
    # determine similarities
    initial_poses, target_poses, predicted_poses = controller.poses
```
