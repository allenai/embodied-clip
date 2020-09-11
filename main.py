"""Training and inference loop for the AI2-THOR object rearrangement task."""

from rearrange_config import Environment
env = Environment(stage='train')
dataset_size = len(env.scenes) * env.shuffles_per_scene

for i_episode in range(dataset_size):
    # walkthrough the target configuration
    for t_step in range(1000):
        rgb_observation = env.last_event.frame

        # START replace with your walkthrough action
        env.action_space.execute_random_action()
        # END replace with your walkthrough action

    # unshuffle to recover the target configuration
    env.shuffle()
    for t_step in range(1000):
        rgb_observation = env.last_event.frame

        # START replace with your unshuffle action
        env.action_space.execute_random_action()
        # END replace with your unshuffle action

    # evaluation
    score = env.evaluate(*env.poses)
    env.reset()  # prepare next episode
