from rearrange_config import Controller
controller = Controller(stage='train')
for i_episode in range(20):
    # walkthrough
    for t_step in range(500):
        rgb_observation = controller.last_event.frame
        controller.action_space.execute_random_action()
    controller.shuffle()
    # unshuffle
    for t_step in range(500):
        rgb_observation = controller.last_event.frame
        controller.action_space.execute_random_action()
    # determine similarities
    initial_poses, target_poses, predicted_poses = controller.poses
    controller.reset()
