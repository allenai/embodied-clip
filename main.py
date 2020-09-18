"""Training and inference loop for the AI2-THOR object rearrangement task."""

from rearrange_config import Environment
env = Environment(
    stage='train',  # or 'val'
    mode='default',  # or 'easy'
    render_instance_masks=False  # only in easy mode
)
dataset_size = len(env.scenes) * env.shuffles_per_scene

for i_episode in range(dataset_size):
    # walkthrough the goal configuration
    for t_step in range(1000):
        rgb, depth, masks = env.observation

        # START replace with your walkthrough action
        env.action_space.execute_random_action()
        # END replace with your walkthrough action

        # only True if agent calls env.done()
        if env.agent_signals_done:
            break

    # unshuffle to recover the goal configuration
    env.shuffle()
    for t_step in range(1000):
        rgb, depth, masks = env.observation

        # START replace with your unshuffle action
        env.action_space.execute_random_action()
        # END replace with your unshuffle action

        # only True if agent calls env.done()
        if env.agent_signals_done:
            break

    # evaluation
    score = env.evaluate(*env.poses)
    env.reset()  # prepare next episode
