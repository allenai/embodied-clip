from rearrange_config import Controller
controller = Controller(stage='train')
dataset_size = len(controller.scenes) * controller.shuffles_per_scene

for i_episode in range(dataset_size):
    # walkthrough the target configuration
    for t_step in range(1000):
        rgb_observation = controller.last_event.frame

        ### START replace with your walkthrough action
        controller.action_space.execute_random_action()
        ### END replace with your action

    # unshuffle to recover the target configuration
    controller.shuffle()
    for t_step in range(1000):
        rgb_observation = controller.last_event.frame

        ### START replace with your unshuffle action
        controller.action_space.execute_random_action()
        ### END replace with your action

    # evaluation
    score = controller.evaluate(*controller.poses)
    controller.reset()  # prepare next episode
