> This repository contains **starter code** for the **object rearrangement task** within AI2-THOR.

<br>

<img src="https://ai2thor.allenai.org/docs/assets/rearrangement/cover.svg" alt="Object Rearrangement Example" width="100%">

## Task Description

The task involves **rearranging and modifying objects** randomly placed in a household. Here the agent:
1. Walks-through the scene with the target objects configured.
2. Resets to its starting position, but the object states around have changed.

> Changes to an object's state may include changes to its position, rotation or openness.

The agent's goal is to **recover the initial configuration** of the scene.

Some of the **key challenges** of performing this task include:
* Identifying which objects have changed.
* Recalling the state of all objects.
* Multi-step reasoning, where manipulating an object might require moving a blocking object.

## Files

This repo consists of the following files:

- **main.py**. Provides a starting snippet to easily set up the task, utilizing _rearrange_config.py_.
- **data/train.json**. Scene configuration data for 80 iTHOR scenes. Within each scene, there are 50 different scene rearrangements tasks. Each rearrangement changes the state of between 1 and 5 objects.
- **data/val.json**. Scene configuration data for 20 unique iTHOR scenes. None of these scenes overlap with _train.json_. Within each scene, there are also 50 different scene rearrangements tasks with each rearrangement changing the state of between 1 and 5 objects.
- **rearrange_config.py**. A helper file that parses the data and configures the objects for each rearrangement. It also provides the goal state of each object.

## Python Setup

### Installation

Each of the actions supports typing within Python, so we require the use of **Python 3.6+**.

To ensure reproducible results, we're restricting all users to use the exact **same version of AI2-THOR**, which can be installed via pip with:

```bash
pip install ai2thor==2.4.12 scipy==1.5.2
```

### main.py

In the `main.py` file, you will find:

```python
from rearrange_config import Controller
controller = Controller(stage='train')
dataset_size = len(controller.scenes) * controller.shuffles_per_scene
for i_episode in range(dataset_size):
    # walkthrough
    for t_step in range(500):
        rgb_observation = controller.last_event.frame
        controller.action_space.execute_random_action()
    controller.shuffle()
    # unshuffle
    for t_step in range(500):
        rgb_observation = controller.last_event.frame
        controller.action_space.execute_random_action()
    score = controller.evaluate(*controller.poses)
    controller.reset()  # prepare next episode
```

**Validation data.** To use the validation data, initialize the controller with:

```python
controller = Controller(stage='val')
```

### Action Space

**Action space property.** Both the _walkthrough_ and the _unshuffling_ phases have their own `ActionSpace` accessible with:

```python
controller.action_space
```

**Walkthrough ActionSpace.** If the controller is currently in the walkthrough phase (i.e., shuffle has not yet been called on the episode), then the action space will consist of:

```python
ActionSpace(
  # move the agent by 0.25 meters
  move_ahead(),
  move_right(),
  move_left(),
  move_back(),

  # rotate the agent by 30 degrees
  rotate_right(),
  rotate_left(),

  # change the agent's height
  stand(),
  crouch(),

  # turn the agent's head by 30 degrees
  look_up(),
  look_down(),

  # agent's done signal
  done()
)
```

**Unshuffling ActionSpace.** If the controller is currently in the unshuffling phase, then the action space will consist of several additional _interactible_ actions:

```python
ActionSpace(
  # move the agent by 0.25 meters
  move_ahead(),
  move_right(),
  move_left(),
  move_back(),

  # rotate the agent by 30 degrees
  rotate_right(),
  rotate_left(),

  # change the agent's height
  stand(),
  crouch(),

  # turn the agent's head by 30 degrees
  look_up(),
  look_down(),

  # agent's done signal
  done(),

  # object interaction
  open_object(
    x: float(low=0, high=1),
    y: float(low=0, high=1),
    openness: float(low=0, high=1)),
  pickup_object(
    x: float(low=0, high=1),
    y: float(low=0, high=1)),
  push_object(
    x: float(low=0, high=1),
    y: float(low=0, high=1),
    rel_x_force: float(low=-0.5, high=0.5),
    rel_y_force: float(low=-0.5, high=0.5),
    rel_z_force: float(low=-0.5, high=0.5),
    force_magnitude: float(low=0, high=1)),

  # held object interaction
  move_held_object(
    x_meters: float(low=-0.5, high=0.5),
    y_meters: float(low=-0.5, high=0.5),
    z_meters: float(low=-0.5, high=0.5)),
  rotate_held_object(
    x: float(low=-0.5, high=0.5),
    y: float(low=-0.5, high=0.5),
    z: float(low=-0.5, high=0.5)),
  drop_held_object()
)
```

**(x, y).** Interacting with an object requires targeting that object. We use `x` and `y` coordinates between [0:1] to target each object, based on the _last RGB image frame_ from the agent's camera. The `x` and `y` coordinates correspond to the relative position of the target object along the horizontal and vertical image axes, respectively. An example of targeting 2 different pickupable objects in the same frame follows:

<img src="https://ai2thor.allenai.org/docs/assets/rearrangement/coordinates.svg" alt="Object Rearrangement Example" width="50%">

**Parameter Scales.** As shown in unshuffle's ActionSpace, all parameters have been scaled between 0 and 1. For `rotate_held_object`, 0.5 corresponds to 90 degrees and -0.5 corresponds to -90 degrees. For `push_object`, a `force_magnitude` of 1 corresponds to 50 newtons of force, which should be sufficient to reasonably move any pickupable object.

**Move held object clipping.** The action `move_held_object` will clip the upper bound at 0.5 meters corresponding to how much a hand held object can move in a single step.

**Random actions.** Demonstrated in `main.py`, randomly execute actions in the action space with:

```python
controller.action_space.execute_random_action()
```

**Specific actions.** Actions can be executed by calling the action from the controller, as in:

```python
controller.move_ahead()
controller.pickup_object(x=0.64, y=0.40)
```

### Object Poses

**Accessing object poses.** After the agent is done both the walkthrough and reshuffling phase, it can access the poses of each object with:

```python
initial_poses, target_poses, predicted_poses = controller.poses
```

**Reading an object's pose.** Here, `initial_poses`, `target_poses`, and `predicted_poses` evaluate to a _list of dictionaries_ and are defined as:

- **initial_poses.** The list of object poses if the agent were to do nothing to the environment during the _unshuffling_ phase.
- **target_poses.** The list of object poses that the agent sees during the walkthrough phase.
- **predicted_poses.** The list of object poses _after_ the agent makes all its changes to the environment during the _unshuffling_ phase.

Each dictionary is an _object's pose_ in the following form:

```js
{
    'type': 'Microwave',
    'position': {
        'x': 1.93299961, 'y': 0.8996917, 'z': -0.766998768},
    'rotation': {
        'x': 0.000180735427, 'y': 270.000183, 'z': 3.77623081e-07},
    'openness': 0.0,
    'is_broken': False,
    'bounding_box': [
        [1.76262379, 0.899691164, -0.453049064],
        [1.76262176, 0.899691164, -1.07892561],
        [2.11550927, 0.8996923, -1.07892668],
        [2.11551118, 0.8996923, -0.4530502],
        [1.76262271, 1.25899935, -0.453049064],
        [1.76262069, 1.25899935, -1.07892561],
        [2.115508, 1.25900042, -1.07892668],
        [2.11551, 1.25900042, -0.4530502]]
}
```

**Matching objects across poses.** Across `initial_poses`, `target_poses`, and `predicted_poses`, the _ith entry_ in each list will _always_ correspond to the same object across each pose list. So, `initial_poses[5]` will refer to the same object as `target_poses[5]` and `predicted_poses[5]`. Most scenes have around 70 objects, among which, 10 to 20 are pickupable by the agent.

**Pose keys:**

- **openness.** For objects where the openness value does not fit (e.g., Bowl, Spoon), the openess value is `None`.
- **bounding_box.** Bounding boxes are only given for moveable objects, where the set of moveable objects may consist of couches or chairs, that are not necessarily pickupable. For pickupable objects, the `bounding_box` is aligned to the object's relative axes. For moveable objects that are non-pickupable, the
- **is_broken.** No object's initial pose or target pose will ever require breaking an object. But, if the agent decides to pick up an object, and drop it on a hard surface, it's possible that the object can break.

### Evaluation

**Evaluation function.** To evaluate a single episode call:

```python
episode_score = controller.evaluate(
    initial_poses,
    target_poses,
    predicted_poses)
```

**Score calculation.** The episode's score ranges between [0:1] and is calculated as follows:

1. If any predicted object is broken, return 0.
2. Otherwise if any non-shuffled object is out of place, return 0.
3. Otherwise return the average number of successfully unshuffled objects.

For steps 2 and 3, an object is considered in-place/unshuffled if it satisfies all of the following:

1. **Openness.** It's `openness` between its target pose and predicted pose is off by less than 20 degrees. The openness check is only applied to objects that can open.
2. **Position and Rotation.** The object's 3D bounding box from its target pose and the predicted pose must have an IoU over 0.5. The positional check is only relevant to object's that can move.
