"""Inference loop for the AI2-THOR object rearrangement task."""
from allenact.utils.misc_utils import NumpyJSONEncoder

from baseline_configs.one_phase.one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)
from baseline_configs.two_phase.two_phase_rgb_base import (
    TwoPhaseRGBBaseExperimentConfig,
)
from rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask

# First let's generate our task sampler that will let us run through all of the
# data points in our training set.

task_sampler_params = TwoPhaseRGBBaseExperimentConfig.stagewise_task_sampler_args(
    stage="train", process_ind=0, total_processes=1,
)
two_phase_rgb_task_sampler: RearrangeTaskSampler = TwoPhaseRGBBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,  # cache used for efficiency during training, should be True during inference
    only_one_unshuffle_per_walkthrough=True,  # used for efficiency during training, should be False during inference
    epochs=1,
)

how_many_unique_datapoints = two_phase_rgb_task_sampler.total_unique
num_tasks_to_do = 5

print(
    f"Sampling {num_tasks_to_do} tasks from the Two-Phase TRAINING dataset"
    f" ({how_many_unique_datapoints} unique tasks) and taking random actions in them. "
)

for i_task in range(num_tasks_to_do):
    print(f"\nStarting task {i_task}")

    walkthrough_task = two_phase_rgb_task_sampler.next_task()
    print(
        f"Sampled task is from the "
        f" '{two_phase_rgb_task_sampler.current_task_spec.stage}' stage and has"
        f" unique id '{two_phase_rgb_task_sampler.current_task_spec.unique_id}'"
    )

    assert isinstance(walkthrough_task, WalkthroughTask)

    # Take random actions in the walkthrough task until the task is done
    while not walkthrough_task.is_done():
        observations = walkthrough_task.get_observations()

        # Take a random action
        action_ind = walkthrough_task.action_space.sample()
        if walkthrough_task.num_steps_taken() % 10 == 0:
            print(
                f"Walkthrough phase (step {walkthrough_task.num_steps_taken()}):"
                f" taking action {walkthrough_task.action_names()[action_ind]}"
            )
        walkthrough_task.step(action=action_ind)

    # Get the next task from the task sampler, this will be the task
    # of rearranging the environment so that it is back in the same configuration as
    # it was during the walkthrough task.
    unshuffle_task: UnshuffleTask = two_phase_rgb_task_sampler.next_task()

    while not unshuffle_task.is_done():
        observations = unshuffle_task.get_observations()

        # Take a random action
        action_ind = unshuffle_task.action_space.sample()
        if unshuffle_task.num_steps_taken() % 10 == 0:
            print(
                f"Unshuffle phase (step {unshuffle_task.num_steps_taken()}):"
                f" taking action {unshuffle_task.action_names()[action_ind]}"
            )
        unshuffle_task.step(action=action_ind)

    print(f"Both phases complete, metrics: '{unshuffle_task.metrics()}'")

print(f"\nFinished {num_tasks_to_do} Two-Phase tasks.")
two_phase_rgb_task_sampler.close()


# Now let's create a One Phase task sampler on the validation dataset.
task_sampler_params = OnePhaseRGBBaseExperimentConfig.stagewise_task_sampler_args(
    stage="valid", process_ind=0, total_processes=1,
)
one_phase_rgb_task_sampler: RearrangeTaskSampler = (
    OnePhaseRGBBaseExperimentConfig.make_sampler_fn(
        **task_sampler_params, force_cache_reset=False, epochs=1,
    )
)

how_many_unique_datapoints = one_phase_rgb_task_sampler.total_unique
print(
    f"\n\nSampling {num_tasks_to_do} tasks from the One-Phase VALIDATION dataset"
    f" ({how_many_unique_datapoints} unique tasks) and taking random actions in them. "
)

for i_task in range(num_tasks_to_do):
    print(f"\nStarting task {i_task}")

    # Get the next task from the task sampler, for One Phase Rearrangement
    # there is only the unshuffle phase (walkthrough happens at the same time implicitly).
    unshuffle_task: UnshuffleTask = one_phase_rgb_task_sampler.next_task()
    print(
        f"Sampled task is from the "
        f" '{one_phase_rgb_task_sampler.current_task_spec.stage}' stage and has"
        f" unique id '{one_phase_rgb_task_sampler.current_task_spec.unique_id}'"
    )

    while not unshuffle_task.is_done():
        observations = unshuffle_task.get_observations()

        # Take a random action
        action_ind = unshuffle_task.action_space.sample()

        if unshuffle_task.num_steps_taken() % 10 == 0:
            print(
                f"Unshuffle phase (step {unshuffle_task.num_steps_taken()}):"
                f" taking action {unshuffle_task.action_names()[action_ind]}"
            )
        unshuffle_task.step(action=action_ind)

    print(f"Both phases complete, metrics: '{unshuffle_task.metrics()}'")

one_phase_rgb_task_sampler.close()

print(f"\nFinished {num_tasks_to_do} One-Phase tasks.")


# When submitting to the leaderboard we will expect you to have evaluated your model on (1) a subset of the
# train set, (2) a subset of the train_unseen, (3) the validation set, and (4) the test set. Running each of these
# evaluations separately is a bit tedious and so we provide a "combined" dataset that combine the above four
# collections together and allows for running through each of them sequentially.
#
# In the following we show how you can iterate through the combined dataset and how we expect your
# agent's results to be saved (see `my_leaderboard_submission` below) before they can be submitted
# to the leaderboard. In practice, sequentially evaluating your agent on each task might be quite slow
# and we recommend paralleling your evaluation. Note that this is done automatically if you run your inference
# using AllenAct, see the last section of our README for details on how this can be done (note that this
# requires that your model/agent is compatible with AllenAct, this is easiest if you trained your agent with
# AllenAct initially).
task_sampler_params = OnePhaseRGBBaseExperimentConfig.stagewise_task_sampler_args(
    stage="combined", process_ind=0, total_processes=1,
)
one_phase_rgb_combined_task_sampler: RearrangeTaskSampler = (
    OnePhaseRGBBaseExperimentConfig.make_sampler_fn(
        **task_sampler_params, force_cache_reset=True, epochs=1,
    )
)

how_many_unique_datapoints = one_phase_rgb_combined_task_sampler.total_unique
print(
    f"\n\nSampling {num_tasks_to_do} tasks from the One-Phase COMBINED dataset"
    f" ({how_many_unique_datapoints} unique tasks) and taking random actions in them. "
)

my_leaderboard_submission = {}
for i_task in range(num_tasks_to_do):
    print(f"\nStarting task {i_task}")

    # Get the next task from the task sampler, for One Phase Rearrangement
    # there is only the unshuffle phase (walkthrough happens at the same time implicitly).
    unshuffle_task: UnshuffleTask = one_phase_rgb_combined_task_sampler.next_task()
    print(
        f"Sampled task is from the "
        f" '{one_phase_rgb_combined_task_sampler.current_task_spec.stage}' stage and has"
        f" unique id '{one_phase_rgb_combined_task_sampler.current_task_spec.unique_id}'"
    )

    while not unshuffle_task.is_done():
        observations = unshuffle_task.get_observations()

        # Take a random action
        action_ind = unshuffle_task.action_space.sample()

        if unshuffle_task.num_steps_taken() % 10 == 0:
            print(
                f"Unshuffle phase (step {unshuffle_task.num_steps_taken()}):"
                f" taking action {unshuffle_task.action_names()[action_ind]}"
            )
        unshuffle_task.step(action=action_ind)

    metrics = unshuffle_task.metrics()
    print(f"Both phases complete, metrics: '{metrics}'")

    task_info = metrics["task_info"]
    del metrics["task_info"]
    my_leaderboard_submission[task_info["unique_id"]] = {**task_info, **metrics}

# Example of saving a gzip'ed file that can be submitted to the leaderboard. Note that we're only
# iterating over `num_tasks_to_do` datapoints in the above loop, to actually make a submission you'd
# have to iterate over all of them.
import json
import gzip
import os

save_path = "/YOUR/FAVORITE/SAVE/PATH/submission.json.gz"
if os.path.exists(os.path.dirname(save_path)):
    print(f"Saving example submission file to {save_path}")
    submission_json_str = json.dumps(my_leaderboard_submission, cls=NumpyJSONEncoder)
    with gzip.open(save_path, "w") as f:
        f.write(submission_json_str.encode("utf-8"))
else:
    print(
        f"If you'd like to save an example leaderboard submission, you'll need to edit"
        "`/YOUR/FAVORITE/SAVE/PATH/` so that it references an existing directory."
    )

one_phase_rgb_combined_task_sampler.close()

print(f"\nFinished {num_tasks_to_do} One-Phase tasks.")
