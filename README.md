<img src="https://ai2thor.allenai.org/docs/assets/rearrangement/cover.svg" alt="Object Rearrangement Example" width="100%">

# 📝 Task Description

**Overview 🤖.** The task involves moving and modifying randomly placed objects within a room so a goal configuration is obtained. There are 2 phases:

1. **Walkthrough 👀.** The agent walks around the room and observes the objects in their ideal goal state.
2. **Unshuffle 🏋.** After the walkthrough phase, we randomly change between 1 to 5 objects in the room. The agent's goal is to identify which objects have changed and reset those objects to their observed state from the walkthrough phase. Changes to an object's state may include changes to its position, orientation, or openness.

# 📁 Files

```bash
git clone https://github.com/allenai/ai2thor-rearrangement.git
```

**main.py 👈.** Provides a starting snippet to easily set up the task, utilizing `rearrange_config.py`. This is intended to be the only file that you modify.

**Static setup files 👊.** These files help execute actions and load the scene:

- `rearrange_config.py` parses the data and configures the objects for each rearrangement. It also provides the goal state of each object.
- `data/train.json` stores scene configuration data for 80 iTHOR scenes. Within each scene, there are 50 different scene rearrangement tasks. Each rearrangement changes the state of between 1 and 5 objects.
- `data/val.json` stores scene configuration data for 20 unique iTHOR scenes. None of these scenes overlap with <span class="chillMono">train.json</span>. Within each scene, there are also 50 different scene rearrangement tasks with each rearrangement changing the state of between 1 and 5 objects.

# 🐍 Python Setup

## 💻 Installation

```bash
pip install -r requirements.txt
```

**Python 3.6+ 🐍.** Each of the actions supports `typing` within <span class="chillMono">Python</span>.

**AI2-THOR 2.4.20 🧞.** To ensure reproducible results, we're restricting all users to use the exact same version of <span class="chillMono">AI2-THOR</span>.

**SciPy 🧑‍🔬.** We utilize <span class="chillMono">SciPy</span> for evaluation. It helps calculate the IoU between 3D bounding boxes.

## ➰ Training and Inference Loop

**Lightweight setup ✨.** In `main.py`, you will find the code to get started:

```python
from rearrange_config import Environment
env = Environment(
    stage='train',  # or 'val'
    mode='default',  # or 'easy'
    render_depth=True
)
dataset_size = len(env.scenes) * env.shuffles_per_scene

for i_episode in range(dataset_size):
    # walkthrough the goal configuration
    for t_step in range(1000):
        rgb, depth = env.observation

        # START replace with your walkthrough action
        env.action_space.execute_random_action()
        # END replace with your walkthrough action

        # only True if agent calls env.done()
        if env.agent_signals_done:
            break

    # unshuffle to recover the goal configuration
    env.shuffle()
    for t_step in range(1000):
        rgb, depth = env.observation

        # START replace with your unshuffle action
        env.action_space.execute_random_action()
        # END replace with your unshuffle action

        # only True if agent calls env.done()
        if env.agent_signals_done:
            break

    # evaluation
    score = env.evaluate(*env.poses)
    env.reset()  # prepare next episode
```

**Number of objects changed 🔢.** For each episode, between 1 and 5 objects have a state change from their initial pose to their goal pose. To access the number of objects that change in a particular episode, call:

```python
env.object_change_n
```

**Mode 🎁.** To help make the task more achievable, we provide an `easy` mode. This mode only allows users to call `env.magic_drop_held_object()`.

**Scene 🛁.** To access the name of the scene, call:

```python
env.scene
```

**Stop 🛑.** To terminate the current environment, call:

```python
env.stop()
```

This is particularly useful if you want to use multiple initialization parameters within the same script.

## 🖼️ Observations

For both the walkthrough and unshuffle phases, the agent only recieves RGB-D observations, accessible at each time step with:

```python
rgb, depth = env.observation
```

<p float="left">
    <img src="https://ai2thor.allenai.org/docs/assets/rearrangement/obs.png" alt="POV Agent Image" width="45%">
    <img src="https://ai2thor.allenai.org/docs/assets/rearrangement/depth.svg" alt="Depth Agent Image" width="54%">
</p>

**RGB image 📷.** The RGB image is a `300x300x3` NumPy array from the agent's eye-level camera. All values are stored as integers between `[0:255]`.

**Depth image 📸.** The depth image is a `300x300` NumPy array from the agent's eye-level camera. We provide unnormalized values, scaled to the meter distance from the agent.

## 🎮 Actions

### 🧑‍🚀 Action Space

The `ActionSpace` for both the walkthrough and the unshuffling phases is accessible with:

```python
env.action_space
```

The actions for the walkthrough 👀 phase and the unshuffling phase 🏋 are shown below.

**1. Move ahead ☝.**

```python
env.move_ahead()
```

Attempts to move the agent ahead by 0.25 meters.

**2. Move left 👈️.**

```python
env.move_left()
```

Attempts to move the agent left by 0.25 meters.

**3. Move right 👉.**

```python
env.move_right()
```

Attempts to move the agent right by 0.25 meters.

**4. Move back 👇.**

```python
env.move_back()
```

Attempts to move the agent back by 0.25 meters.

**5. Rotate right ️↩️.**

```python
env.rotate_right()
```

Attempts to rotate the agent right by 30 degrees.

**6. Rotate left ↪️.**

```python
env.rotate_left()
```

Attempts to rotate the agent left by 30 degrees.

**7. Stand 🧍.**

```python
env.stand()
```

Attempts to stand the agent from a crouching position.

**8. Crouch 🧎.**

```python
env.crouch()
```

Attempts to crouche the agent from a standing position.

**9. Look up 🙄.**

```python
env.look_up()
```

Attempts to rotate the agent’s head upward by 30 degrees. The maximum upward angle agent can look is 30 degrees.

**10. Look down 😔.**

```python
env.look_down()
```

Attempts to rotate the agent’s head downward by 30 degrees. The maximum downward angle agent can look is 60 degrees.

**11. Done ✅.**

```python
env.done()
```

Agent’s signal that it has completed the current phase and is ready to move on. The result of this action is that `env.agent_signals_done` becomes `True`.

**12. Open object 📖️.**

> Unshuffle phase only.

```python
env.open_object(
    x: float(low=0, high=1),
    y: float(low=0, high=1),
    openness: float(low=0, high=1))
```

Attempts to open the object at [🎯Target Point](#-target-point) `(x, y)` to `openness` percent.

**13. Pickup object 🏋.**

> Unshuffle phase only.

```python
env.pickup_object(
    x: float(low=0, high=1),
    y: float(low=0, high=1))
```

Attempts to pick up the object at [🎯Target Point](#-target-point) `(x, y)`.

**14. Push object 📌.**

> Unshuffle phase only.

```python
env.push_object(
    x: float(low=0, high=1),
    y: float(low=0, high=1),
    rel_x_force: float(low=-0.5, high=0.5),
    rel_y_force: float(low=-0.5, high=0.5),
    rel_z_force: float(low=-0.5, high=0.5),
    force_magnitude: float(low=0, high=1))
```

Attempts to push the object at [🎯Target Point](#-target-point) `(x, y)`. Here, the relative forces (`rel_x_force`, `rel_y_force`, `rel_z_force`) provide the directional force vector. A `force_magnitude` of 1 corresponds to 50 newtons of force, which should be sufficient to reasonably move any pickupable object.

**15. Move held object 👊.**

> Unshuffle phase only.

```python
env.move_held_object(
    x_meters: float(low=-0.5, high=0.5),
    y_meters: float(low=-0.5, high=0.5),
    z_meters: float(low=-0.5, high=0.5))
```

Attempts to move the object in the agent's hand. Here, the `y` coordinate is up and down.

> The maximum amount a hand will move in a single time step is 0.5 meters. If the specified L2 magnitude from all 3 directions is over this mark, the object will simply move 0.5 meters in the given direction.

**16. Rotate held object 👋️.**

> Unshuffle phase only.

```python
env.rotate_held_object(
    x: float(low=-0.5, high=0.5),
    y: float(low=-0.5, high=0.5),
    z: float(low=-0.5, high=0.5))
```

Attempts to rotate the object in the agent's hand. Here, 0.5 corresponds to 90 degrees and -0.5 corresponds to -90 degrees.

**17. Drop held object ✋.**

> Unshuffle phase only.

```python
env.drop_held_object()
```

Drops the object in the hand of an agent, if the hand is holding an object.

> Dropping some objects may cause them to break, which we consider a [failed unshuffling](#-evaluation).

**18. Magic drop held object 🔮.**

> Unshuffle phase and 🟢 easy mode only.

```python
env.magic_drop_held_object()
```

Perfectly drops the object in the agent's hand to its goal pose, if the agent is within 1.5 meters of the goal pose of the object, and looking in its direction. Otherwise, the object is dropped normally.

<br>

### 🎯 Target Point

Interacting with an object requires targeting that object. We use `x` and `y` coordinates between [0:1] to target each object, based on the _last RGB image frame_ from the agent's camera.

The `x` and `y` coordinates correspond to the relative position of the target object along the horizontal and vertical image axes, respectively. An example of targeting 2 different pickupable objects in the same frame follows:

<img src="https://ai2thor.allenai.org/docs/assets/rearrangement/coordinates.svg" alt="Object Rearrangement Example" width="50%">

### 👻 Random

As demonstrated in `main.py`, we can randomly execute actions in the `ActionSpace` with:

```python
env.action_space.execute_random_action()
```

## 🪑 Object Poses

**Accessing object poses 🧘.** After the agent is done both the walkthrough and unshuffle phase, it can access the poses of each object with:

```python
initial_poses, goal_poses, predicted_poses = env.poses
```

**Reading an object's pose 📖.** Here, `initial_poses`, `goal_poses`, and `predicted_poses` evaluate to a _list of dictionaries_ and are defined as:

- `initial_poses` stores a list of object poses if the agent were to do nothing to the <span class="chillMono">env</span> during the _unshuffling_ phase.
- `goal_poses` stores a list of object poses that the agent sees during the walkthrough phase.
- `predicted_poses` stores a list of object poses _after_ the agent makes all its changes to the <span class="chillMono">env</span> during the _unshuffling_ phase.

Each dictionary contains the object's pose in the following form:

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

**Matching objects across poses 🤝.** Across `initial_poses`, `goal_poses`, and `predicted_poses`, the _ith entry_ in each list will _always_ correspond to the same object across each pose list. So, `initial_poses[5]` will refer to the same object as `goal_poses[5]` and `predicted_poses[5]`. Most scenes have around 70 objects, among which, 10 to 20 are pickupable by the agent.

**Pose keys 🔑.**

- `openness` specifies the <span class="chillMono">[0:1]</span> percentage that an object is opened. For objects where the <span class="chillMono">openness</span> value does not fit (e.g., <span class="chillMono">Bowl</span>, <span class="chillMono">Spoon</span>), the <span class="chillMono">openness</span> value is <span class="chillMono">None</span>.
- `bounding_box` is only given for moveable objects, where the set of moveable objects may consist of couches or chairs, that are not necessarily pickupable. For pickupable objects, the bounding_box is aligned to the object's relative axes. For moveable objects that are non-pickupable, the object is aligned to the global axes.
- `is_broken` states if the object broke from the agent's actions during the unshuffling phase. The initial pose or goal pose for each object will never be broken. But, if the agent decides to pick up an object, and drop it on a hard surface, it's possible that the object can break.

## 🏆 Evaluation

**Evaluation function 📏.** To evaluate a single episode, call:

```python
episode_score = env.evaluate(
    initial_poses,
    goal_poses,
    predicted_poses)
```

**Score calculation 💯.** The episode's score ranges between `[0:1]` and is calculated as follows:

1. ❌ If any predicted object is broken, return 0.
2. ❌ Otherwise if any non-shuffled object is out of place, return 0.
3. ✔️ Otherwise return the average number of successfully unshuffled objects.

For steps 2 and 3, a predicted object is considered successfully in-place/unshuffled if it satisfies both of the following:

1. **Openness 📖.** The openness between its goal state and predicted state is off by less than 20 percent. The openness check is only applied to objects that can open.
2. **Position 📍 and Rotation 🙃.** The object's 3D bounding box from its goal pose and the predicted pose must have an IoU over 0.5. The positional check is only relevant to objects that can move.
