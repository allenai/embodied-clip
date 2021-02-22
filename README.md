# 2021 AI2-THOR Rearrangement Challenge

Welcome to the 2021 AI2-THOR Rearrangement Challenge hosted at the
[CVPR'21 Embodied-AI Workshop](https://embodied-ai.org/).
The goal of this challenge is to build a model/agent that move objects in a room
to restore them to a given initial configuration. Please follow the instructions below
to get started.



### Contents
<!--
[TOC]
with open("README.md", "r") as f:
    a = markdown.markdown(f.read(), extensions=["toc"])
    print(a[:a.index("</div>") + 6])
-->
<div class="toc">
<ul>
<li><a href="#-installation">💻 Installation</a></li>
<li><a href="#-rearrangement-task-description">📝 Rearrangement Task Description</a></li>
<li><a href="#-challenge-tracks-and-datasets">🛤️ Challenge Tracks and Datasets</a><ul>
<li><a href="#%EF%B8%8F%EF%B8%8F-the-1--and-2-phase-tracks">☝️+✌️ The 1- and 2-Phase Tracks</a></li>
<li><a href="#-datasets">📊 Datasets</a></li>
</ul>
</li>
<li><a href="#-submitting-to-the-leaderboard">🛤️ Submitting to the Leaderboard</a></li>
<li><a href="#-allowed-observations">🖼️ Allowed Observations</a></li>
<li><a href="#-allowed-actions">🏃 Allowed Actions</a></li>
<li><a href="#%EF%B8%8F-setting-up-rearrangement">🍽️ Setting up Rearrangement</a><ul>
<li><a href="#-learning-by-example">✨ Learning by example</a></li>
<li><a href="#-the-rearrange-thor-environment-class">🌎 The Rearrange THOR Environment class</a></li>
<li><a href="#-the-rearrange-task-sampler-class">🏒 The Rearrange Task Sampler class</a></li>
<li><a href="#-the-walkthrough-task-and-unshuffle-task-classes">🚶🔀 The Walkthrough Task and Unshuffle Task classes</a></li>
</ul>
</li>
<li><a href="#-object-poses">🗺️ Object Poses</a></li>
<li><a href="#-evaluation">🏆 Evaluation</a><ul>
<li><a href="#-when-are-poses-approximately-equal">📏 When are poses (approximately) equal?</a></li>
<li><a href="#-computing-metrics">💯 Computing metrics</a></li>
</ul>
</li>
<li><a href="#-training-baseline-models-with-allenact">🏋 Training Baseline Models with AllenAct</a><ul>
<li><a href="#-pretrained-1-phase-model">💪 Pretrained 1-Phase Model</a></li>
</ul>
</li>
</ul>
</li>
</ul>
</div>

## 💻 Installation

To begin, clone this repository locally
```bash
git clone git@github.com:allenai/ai2thor-rearrangement.git
```
<details>
<summary><b>See here for a summary of the most important files/directories in this repository</b> </summary> 
<p>

Here's a quick summary of the most important files/directories in this repository:
* `example.py` an example script showing how rearrangement tasks can be instantiated for training and
    validation.
* `baseline_configs/`
    - `rearrange_base.py` The base configuration file which defines the challenge
    parameters (e.g. screen size, allowed actions, etc).
    - `one_phase/*.py` - Baseline experiment configurations for the 1-phase challenge track.
    - `two_phase/*.py` - Baseline experiment configurations for the 2-phase challenge track.
    - `walkthrough/*.py` - Baseline experiment configurations if one wants to train the walkthrough
    phase in isolation.
* `rearrange/`
    - `baseline_models.py` - A collection of baseline models for the 1- and 2-phase challenge tasks. These
      Actor-Critic models use a CNN->RNN architecture and can be trained using the experiment configs
      under the `baseline_configs/[one/two]_phase/` directories.
    - `constants.py` - Constants used to define the rearrangement task. These include the step size 
      taken by the agent, the unique id of the the THOR build we use, etc.
    - `environment.py` - The definition of the `RearrangeTHOREnvironment` class that wraps the AI2-THOR
      environment and enables setting up rearrangement tasks.
    - `expert.py` - The definition of a heuristic expert (`GreedyUnshuffleExpert`) which uses privileged 
      information (e.g. the scene graph & knowledge of exact object poses) to solve the rearrangement task.
      This heuristic expert is meant to be used to produce expert actions for use with imitation learning 
      techinques. See the `query_expert` method within the `rearrange.tasks.UnshuffleTask` class for
      an example of how such an action can be generated.
    - `losses.py` - Losses (outside of those provided by AllenAct by default) used to train our
      baseline agents.
    - `sensors.py` - Sensors which provide observations to our agents during training. E.g. the
      `RGBRearrangeSensor` obtains RGB images from the environment and returns them for use by the agent. 
    - `tasks.py` - Definitions of the `UnshuffleTask`, `WalkthroughTask`, and `RearrangeTaskSampler` classes.
      For more information on how these are used, see the [Setting up Rearrangement](#%EF%B8%8F-setting-up-rearrangement) 
      section.
    - `utils.py` - Standalone utility functions (e.g. computing IoU between 3D bounding boxes).

</p>
</details>

You can then install requirements by running
```bash
pip install -r requirements.txt
```
or, if you prefer using conda, we can create a `thor-rearrange` environment with our requirements by running
```bash
export MY_ENV_NAME=thor-rearrange
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
PIP_SRC="${CONDA_BASE}/envs/${MY_ENV_NAME}/pipsrc" conda env create --file environment.yml --name $MY_ENV_NAME
```
<details>
<summary> <b> Why not just run <code>conda env create --file environment.yml --name thor-rearrange</code> by itself? </b></summary> 
<p>

If you were to run `conda env create --file environment.yml --name thor-rearrange` nothing would break
but we have some pip requirements in our `environment.yml` file and, by default, these are saved in 
a local `./src` directory. By explicitly specifying the `PIP_SRC` variable we can have it place these
 pip-installed packages in a nicer (more hidden) location.

</p>
</details>



**Python 3.6+ 🐍.** Each of the actions supports `typing` within <span class="chillMono">Python</span>.

**AI2-THOR 2.7.2 🧞.** To ensure reproducible results, we're restricting all users to use the exact same version of <span class="chillMono">AI2-THOR</span>.

**AllenAct 🏋💪.** We ues the <span class="chillMono">AllenAct</span> reinforcement learning framework 
    for generating baseline models, baseline training pipelines, and for several of their helpful abstractions/utilities.

**SciPy 🧑‍🔬.** We utilize <span class="chillMono">SciPy</span> for evaluation. It helps calculate the IoU between 3D bounding boxes.

## 📝 Rearrangement Task Description

<img src="https://ai2thor.allenai.org/static/0f682c0103df1060810ad214c4668718/06655/rearrange-cover2.jpg" alt="Object Rearrangement Example" width="100%">

**Overview 🤖.** Our rearrangement task involves moving and modifying (i.e. opening/closing) randomly placed objects within a room
to obtain a goal configuration. There are 2 phases:

1. **Walkthrough 👀.** The agent walks around the room and observes the objects in their ideal goal state.
2. **Unshuffle 🏋.** After the walkthrough phase, we randomly change between 1 to 5 objects in the room.
    The agent's goal is to identify which objects have changed and reset those objects to their state from the
    walkthrough phase. Changes to an object's state may include changes to its position, orientation, or openness.
   
## 🛤️ Challenge Tracks and Datasets

### ☝️+✌️ The 1- and 2-Phase Tracks

For this 2021 challenge we have two distinct tracks:

* **1-Phase Track (Easier).** In this track we merge both of the above phases into a single phase. At every step
  the agent obtains observations from the walkthrough (goal) state as well as the shuffled state. This allows the
  agent to directly compare aligned images from the two world-states and thus makes it much easier to determine
  if an object is, or is not, in its goal pose.
* **2-Phase Track (Harder).** In this track, the walkthrough and unshuffle phases occur sequentially and so, once 
  in the unshuffle phase, the agent no longer has any access to the walkthrough state except through any memory
  it has saved.
  
### 📊 Datasets

For this challenge we have four distinct dataset splits: `"train"`, `"train_unseen"`, `"val"`, and `"test"`.
The `train` and `train_unseen` splits use floor plans 1-20, 200-220, 300-320, and 400-420 within AI2-THOR,
the `"val"` split uses floor plans 21-25, 221-225, 321-325, and 421-425, and finally the `"test"` split uses
scenes 26-30, 226-230, 326-330, and 426-430. These dataset splits are stored as the compressed [pickle](https://docs.python.org/3/library/pickle.html)-serialized files
`data/*.pkl.gz`. While you are freely (and encouraged) to enhance the training set as you see fit, you should
never train your agent within any of the test scenes.

For evaluation, your model will need to be evaluated on each of the above splits and the results
submitted to our leaderboard link (see section below). As the `"train"` and `"train_unseen"` sets
are quite large, we do not expect you to evaluate on their entirety. Instead we select ~1000 datapoints
from each of these sets for use in evaluation. For convenience, we provide the `data/combined.pkl.gz`
file which contains the `"train"`, `"train_unseen"`, `"val"`, and `"test"` datapoints that should
be used for evaluation.

| Split        | # Total Episodes | # Episodes for Eval | Path |
| ------------ |:-----:|-----|-----|
| train        | 4000 | 1200 | `data/train.pkl.gz`|
| train_unseen | 3800 | 1140 | `data/train_unseen.pkl.gz`|
| val          | 1000 | 1000 | `data/val.pkl.gz` | 
| test         | 1000 | 1000 | `data/test.pkl.gz` |
| combined     | 4340 | 4340 | `data/combined.pkl.gz` |
  
## 🛤️ Submitting to the Leaderboard

We will be tracking challenge participant entries using the [AI2 Leaderboard](https://leaderboard.allenai.org/).
Submissions will be opened around March 1st and a submission link will be provided in this section along with
instructions on the expected format for submissions.

## 🖼️ Allowed Observations

In both of these tracks, agents should make decisions based off of egocentric sensor readings. The
types of sensors allowed/provided for this challenge include:

<p float="left">
    <img src="https://ai2thor.allenai.org/static/3b1dea7228ed5c3fab03fb5f960173eb/bc8e0/rgb-frame.png" alt="POV Agent Image" width="45%">
    <img src="https://ai2thor.allenai.org/static/73f2a583b1636712a7a7d165ed6d768d/d79bd/depth-frame.jpg" alt="Depth Agent Image" width="54%">
</p>

1. **RGB images** - having shape `224x224x3` and an FOV of 90 degrees.  
2. **Depth maps** - having shape `224x224` and an FOV of 90 degrees.
3. **Perfect egomotion** - We allow for agents to know precisely how far (and in which direction)
    they have moved as well as how many degrees they have rotated. 

While you are absolutely free to use any sensor information you would like during training (e.g.
pretraining your CNN using semantic segmentations from AI2-THOR or using a scene graph to compute
expert actions for imitation learning) such additional sensor information should not
be used at inference time.

## 🏃 Allowed Actions

A total of 82 actions are available to our agents, these include:

**Navigation**
* `Move[Ahead/Left/Right/Back]` - Results in the agent moving 0.25m in the specified direction if doing so would
    not result in the agent colliding with something.
  
* `Rotate[Right/Left]` - Results in the agent rotating 90 degrees clockwise (if `Right`) or counterclockwise (if `Left`).
    This action may fail if the agent is holding an object and rotating would cause the object to collide.
  
* `Look[Up/Down]` - Results in the agent raising or lowering its camera angle by 30 degrees (up to a max of 60 degrees below horizontal and 30 degrees above horizontal).

**Object Interaction**

* `Pickup[OBJECT_TYPE]` - Where `OBJECT_TYPE` is one of the 62 pickupable object types, see `constants.py`.
    This action results in the agent picking up a visible object of type `OBJECT_TYPE` if: (a) the agent is not already
  holding an object, (b) the agent is close enough to the object (within 1.5m), and picking up the object would not
  result in it colliding with objects in front of the agent. If there are multiple objects of type `OBJECT_TYPE`
  then one is chosen at random.
  
* `Open[OBJECT_TYPE]` - Where `OBJECT_TYPE` is one of the 10 opennable object types that are not also
  pickupable, see `constants.py`. If an object whose openness is different from the openness in the goal
  state is visible and within 1.5m of the agent, this object's openness is changed to its value in the goal state.
  
* `PlaceObject` - Results in the agent dropping its held object. If the held object's goal state is visible and within
  1.5m of the agent, it is placed into that goal state. Otherwise, a heuristic is used to place the 
  object on a nearby surface.
  
**Done action**
    
* `Done` - Results in the walkthrough or unshuffle phase immediately terminating.

## 🍽️ Setting up Rearrangement

### ✨ Learning by example

See the `example.py` file for an example of how you can instantiate the 1- and 2-phase
variants of our rearrangement task.

### 🌎 The Rearrange THOR Environment class

The `rearrange.environment.RearrangeTHOREnvironment` class provides a wrapper around the AI2-THOR environment
and is designed to 
1. Make it easy to set up a AI2-THOR scene in a particular state ready for rearrangement.
1. Provides utilities to make it easy to evaluate (see e.g. the `poses` and  `compare_poses` methods)
  how close the current state of the environment is to the goal state.
1. Provide an API with which the agent may interact with the environment.

### 🏒 The Rearrange Task Sampler class

You'll notice that the above `RearrangeTHOREnvironment` is not explicitly instantiated by the `example.py`
script and, instead, we create `rearrange.tasks.RearrangeTaskSampler` objects using the
`TwoPhaseRGBBaseExperimentConfig.make_sampler_fn` and `OnePhaseRGBBaseExperimentConfig.make_sampler_fn`.
This is because the `RearrangeTHOREnvironment` is very flexible and doesn't know anything about
training/validation/test datasets, the types of actions we want our agent to be restricted to use,
or precisely which types of sensor observations we want to give our agents (e.g. RGB images, depth maps, etc).
All of these extra details are managed by the `RearrangeTaskSampler` which iteratively creates new
tasks for our agent to complete when calling the `next_task` method. During training, these new tasks can be sampled
indefinitely while, during validation or testing, the tasks will only be sampled until the validation/test datasets
are exhausted. This sampling is best understood by example so please go over the `example.py` file.

### 🚶🔀 The Walkthrough Task and Unshuffle Task classes

As described above, the `RearrangeTaskSampler` samples tasks for our agent to complete, these tasks correspond
to instantiations of the `rearrange.tasks.WalkthroughTask` and `rearrange.tasks.UnshuffleTask` classes. For the 2-phase
challenge track, the `RearrangeTaskSampler` will first sample a new `WalkthroughTask` after which it will sample a 
corresponding `UnshuffleTask` where the agent must return the objects to their poses at the start of the
`WalkthroughTask`. 

## 🗺️ Object Poses

**Accessing object poses 🧘.** The poses of all objects in the environment can be accessed
using the `RearrangeTHOREnvironment.poses` property, i.e.

```python
unshuffle_start_poses, walkthrough_start_poses, current_poses = env.poses # where env is an RearrangeTHOREnvironment instance  
```

**Reading an object's pose 📖.** Here, `unshuffle_start_poses`, `walkthrough_start_poses`, and `current_poses`
evaluate to a _list of dictionaries_ and are defined as:

- `unshuffle_start_poses` stores a list of object poses if the agent were to do nothing to
  the `env` during the _unshuffling_ phase.
- `walkthrough_start_poses` stores a list of object poses that the agent sees during the walkthrough phase.
- `current_poses` stores a list of object poses in the current state of the environment (i.e. possibly after the
    unshuffle agent makes all its changes to the `env` during the _unshuffling_ phase).

Each dictionary contains the object's pose in a form similar to:

```js
{
    "type": "Candle",
    "position": {
        "x": -0.3012670874595642,
        "y": 0.7431036233901978,
        "z": -2.040205240249634
    },
    "rotation": {
        "x": 2.958569288253784,
        "y": 0.027708930894732475,
        "z": 0.6745457053184509
    },
    "openness": None,
    "pickupable": True,
    "broken": False,
    "objectId": "Candle|-00.30|+00.74|-02.04",
    "name": "Candle_977f7f43",
    "parentReceptacles": [
        "Bathtub|-01.28|+00.28|-02.53"
    ],
    "bounding_box": [
        [-0.27043721079826355, 0.6975823640823364, -2.0129783153533936],
        [-0.3310248851776123, 0.696869969367981, -2.012985944747925],
        [-0.3310534358024597, 0.6999208927154541, -2.072017192840576],
        [-0.27046576142311096, 0.7006332278251648, -2.072009563446045],
        [-0.272365003824234, 0.8614493608474731, -2.0045082569122314],
        [-0.3329526484012604, 0.8607369661331177, -2.0045158863067627],
        [-0.3329811990261078, 0.8637878894805908, -2.063547134399414],
        [-0.27239352464675903, 0.8645002245903015, -2.063539505004883]
    ]
}
```

**Matching objects across poses 🤝.** Across `unshuffle_start_poses`, `walkthrough_start_poses`, and `current_poses`,
    the _ith entry_ in each list will _always_ correspond to the same object across each pose list. 
    So, `unshuffle_start_poses[5]` will refer to the same object as `walkthrough_start_poses[5]` and `current_poses[5]`. 
    Most scenes have around 70 objects, among which, 10 to 20 are pickupable by the agent.

**Pose keys 🔑.**

- `openness` specifies the `[0:1]` percentage that an object is opened. For objects where the `openness` value does not
  fit (e.g., `Bowl`, `Spoon`), the `openness` value is `None`.
- `bounding_box` is only given for moveable objects, where the set of moveable objects may consist of couches or chairs,
  that are not necessarily pickupable. For pickupable objects, the bounding_box is aligned to the object's relative
  axes. For moveable objects that are non-pickupable, the object is aligned to the global axes.
- `broken` states if the object broke from the agent's actions during the unshuffling phase. The initial pose or 
  goal pose for each object will never be broken. But, if the agent decides to pick up an object, and drop it on a hard 
  surface, it's possible that the object can break.

## 🏆 Evaluation

To evaluate the quality of a rearrangement agent we compute several metrics measuring how well the agent
has managed to move objects so that their final poses are (approximately) equal to their goal poses.

### 📏 When are poses (approximately) equal?
Recall that we represent the pose of an object as a combination of its:
1. **Openness 📖.** - A value in [0,1] which measures how far the object has been opened.
2. **Position 📍, Rotation 🙃, and bounding box 📦** - The 3D position, rotation, and bounding box of each object. 
3. **Broken** - A boolean indicating if the object has been broken (all goal object poses are unbroken).

The openness between its goal state and predicted state is off by less than 20 percent. The openness check is only applied to objects that can open.
The object's 3D bounding box from its goal pose and the predicted pose must have an IoU over 0.5. The positional check is only relevant to objects that can move.

To measure if two object poses are approximately equal we use the following criterion:
1. ❌ If any object pose is broken.
1. ❌ If the object is opennable but not pickupable (e.g. a cabinet) and the the openness values
    between the two poses differ by more than 0.2.
1. ❌ The two 3D bounding boxes of pickupable objects have an IoU under 0.5.
1. ✔️ None of the above criteria are met so the poses are not broken, are close in openness values, and have
    sufficiently high IoU.

### 💯 Computing metrics

Suppose that `task` is an instance of an `UnshuffleTask` which your agent has taken
    actions until reaching a terminal state (e.g. either the agent has taken the maximum number of steps or it
    has taken the `"done"` action). Then metrics regarding the agent's performance can be computed by calling
    the `task.metrics()` function. This will return a dictionary of the form
```python
{
    "task_info": {
        "scene": "FloorPlan420",
        "index": 7,
        "stage": "train"
    },
    "ep_length": 176,
    "unshuffle/ep_length": 7,
    "unshuffle/reward": 0.5058389582634852,
    "unshuffle/start_energy": 0.5058389582634852,
    "unshuffle/end_energy": 0.0,
    "unshuffle/prop_fixed": 1.0,
    "unshuffle/prop_fixed_strict": 1.0,
    "unshuffle/num_misplaced": 0,
    "unshuffle/num_newly_misplaced": 0,
    "unshuffle/num_initially_misplaced": 1,
    "unshuffle/num_fixed": 1,
    "unshuffle/num_broken": 0,
    "unshuffle/change_energy": 0.5058464936498058,
    "unshuffle/num_changed": 1,
    "unshuffle/prop_misplaced": 0.0,
    "unshuffle/energy_prop": 0.0,
    "unshuffle/success": 0.0,
    "walkthrough/ep_length": 169,
    "walkthrough/reward": 1.82,
    "walkthrough/num_explored_xz": 17,
    "walkthrough/num_explored_xzr": 46,
    "walkthrough/prop_visited_xz": 0.5151515151515151,
    "walkthrough/prop_visited_xzr": 0.3484848484848485,
    "walkthrough/num_obj_seen": 11,
    "walkthrough/prop_obj_seen": 0.9166666666666666
}
```
Of the above metrics, the most important (those used for comparing models) are
* **Success rate** (`"unshuffle/success"`) - This is the most unforgiving of our metrics and equals 1 if all
  object poses are in their goal states after the unshuffle phase.
* **% Misplaced** (`"unshuffle/prop_misplaced"`) - The above sucess metric is quite strict, requiring exact 
  rearrangement of all objects, and also does not additionally penalize an agent for moving objects
  that should not be moved. This metric equals the number of misplaced objects at the end of the episode divided 
  by the number of misplaced objects at the start of the episode. Note that this metric can be larger than 1
  if the agent, during the unshuffle stage, misplaces more objects than were originally misplaced at the start.
* **% Fixed Strict** (`"unshuffle/prop_fixed_strict"`) - This metric equals 0 if, at the end of the unshuffle task,
  the agent has misplaced any new objects (i.e. it has incorrectly moved an object that started in its correct
  position). Otherwise, if it has not misplaced new objects, then this equals
  (# objects which started in the wrong pose but are now in the correct pose) / (# objects which started in 
  an incorrect pose), i.e. the proportion of objects who had their pose __fixed__.
* **% Energy Remaining** (`"unshuffle/energy_prop"`) - The above metrics do not give any partial credit if, 
  for example, the agent moves an object across a room and towards its goal pose but fails to place it so that
  has a sufficiently high IOU with the goal. To allow for partial credit, we define an energy function 
  `D` that monotonically decreases to 0 as two poses get closer together (see code for full details) and which equals 
  zero if two poses are approximately equal. This metric is then defined as the amount of energy remaining at the end
  of the unshuffle episode divided by the total energy at the start of the unshuffle episode, 
  i.e. equals (sum of energy between all goal/current object poses at end of the unshuffle phase) / 
  (sum of energy between all goal/current object poses at the start of the unshuffle phase).
  

## 🏋 Training Baseline Models with AllenAct

We use the [AllenAct framework](https://www.allenact.org) for training our baseline rearrangement models, 
this framework is automatically installed when [installing the requirements for this project](#installation). 
Let's say you want to train a model for the 1-phase challenge. This can be easily done by running the command 
```bash
allenact -o rearrange_out -b . baseline_configs/one_phase/one_phase_rgb_resnet_dagger.py 
```
This will train (using DAgger, a form of imitation learning) a model which uses a pretrained (with frozen
weights) ResNet18 as the visual backbone that feeds into a recurrent neural network (a GRU) before 
producing action probabilities and a value estimate. Results from this training are then saved to
`rearrange_out` where you can find model checkpoints, tensorboard plots, and configuration files that can
be used if you, in the future, forget precisely what the details of your experiment were.

A similar model can be trained for the 2-phase challenge by running
```bash
allenact -o rearrange_out -b . baseline_configs/two_phase/two_phase_rgb_resnet_ppowalkthrough_ilunshuffle.py
```

### 💪 Pretrained 1-Phase Model

We provide a pretrained baseline model for the 1-phase task. This model can be downloaded from 
[this link](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/rearrangement/one-phase/exp_OnePhaseRGBResNetDagger_40proc__time_2021-02-07_11-25-27__stage_00__steps_000075001830.pt)
and should be placed into the `pretrained_model_ckpts` directory. You can then run inference on this
model using AllenAct using the command
```bash
allenact baseline_configs/one_phase/one_phase_rgb_resnet_dagger.py -c pretrained_model_ckpts/exp_OnePhaseRGBResNetDagger_40proc__time_2021-02-07_11-25-27__stage_00__steps_000075001830.pt -t 2021-02-07_11-25-27
```
this will evaluate this model across all datapoints in the `data/combined.pkl.gz` dataset
which contains data from the `train`, `train_unseen`, `val`, and `test` sets so that
evaluation doesn't have to be run on each set separately.
