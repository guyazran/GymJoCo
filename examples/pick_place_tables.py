import numpy as np

from gymjoco.episode.specs.robot_spec import RobotSpec, AttachmentSpec
from gymjoco.episode.samplers import MultiTaskEpisodeSampler
from gymjoco.episode.specs.scene_spec import SceneSpec
from gymjoco.episode.specs.task_spec import TaskSpec
from gymjoco.env import GymJoCoEnv
from gymjoco.tasks.rearrangement.rearrangement_task import COMRearrangementTask


class RandomClutter(MultiTaskEpisodeSampler):
    def __init__(self, scene, robot, obs_heights, obj_heights):
        super().__init__(scene, robot)
        self.obj_heights = obj_heights
        self.obs_heights = obs_heights

    def _sample_task(self) -> TaskSpec:
        params = dict(obj_poses={}, obs_poses={}, time_limit=5000)

        for name, height in self.obs_heights.items():
            params['obs_poses'][f'obstacle_{name}'] = self.__sample_obj_poses(height)

        for name, height in self.obj_heights.items():
            params['obj_poses'][f'pick_object_{name}'] = self.__sample_obj_poses(height)

        return TaskSpec(cls=COMRearrangementTask, params=params)

    def __sample_obj_poses(self, height):
        start_x, start_y = self.__ring_sample(0.3, 0.7)

        # start_quat = _np.random.uniform(-1, 1, 4)
        # start_quat = start_quat / _np.linalg.norm(start_quat)

        goal_x, goal_y = self.__ring_sample(0.3, 0.7)

        return {
            'start_pose': [start_x, start_y, height] + [1, 0, 0, 0],  # start_quat.tolist(),
            'goal_com': [goal_x, goal_y, height]
        }

    def __ring_sample(self, radius_min, radius_max):
        theta = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(radius_min, radius_max)

        return radius * np.cos(theta), radius * np.sin(theta)


env = GymJoCoEnv(
    episode_sampler=RandomClutter(
        SceneSpec('multiobjworld', init_keyframe='home'),
        RobotSpec('mujoco_menagerie/universal_robots_ur5e/ur5e.xml',
                  attachments=[
                      # AttachmentSpec('mujoco_menagerie/shadow_hand/left_hand.xml',
                      #                base_rot=np.array([np.pi, np.pi/2, 0])),
                      AttachmentSpec('mujoco_menagerie/robotiq_2f85/2f85.xml')
                  ]
                  ),
        obs_heights=dict(zip(range(4), [0.3, 0.3, 0.3, 0.3])),
        obj_heights={'red': 0.05, 'yellow': 0.05, 'cyan': 0.05}
    ),
    render_mode='human',
)

N_EPISODES = 100

for _ in range(N_EPISODES):
    obs, info = env.reset()
    done = False
    i = 0
    while not done:
        i += 1
        action = env.action_space.sample()

        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        env.render()

env.close()
