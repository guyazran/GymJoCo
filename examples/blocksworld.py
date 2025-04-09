import os

from tqdm import tqdm

import gymjoco
from gymjoco.common.defs import cfg_keys
from gymjoco.episode import *
from gymjoco.tasks.null_task import NullTask

scene = SceneSpec(
    '3tableworld',
     objects=(
         ObjectSpec('cubeA', base_pos=[0.14, 0.5, 0.705], base_joints=JointSpec('free')),
         ObjectSpec('cubeB', base_pos=[-0.14, 0.5, 0.705], base_joints=JointSpec('free')),
         ObjectSpec('cubeC', base_pos=[0.14, 0.75, 0.705], base_joints=JointSpec('free')),
         ObjectSpec('cubeD', base_pos=[-0.14, 0.75, 0.705], base_joints=JointSpec('free')),
         ObjectSpec('cubeE', base_pos=[0.14, -0.5, 0.705], base_joints=JointSpec('free')),
         ObjectSpec('cubeF', base_pos=[-0.14, -0.5, 0.705], base_joints=JointSpec('free')),
     ),
    render_camera='rightangleview',
    init_keyframe='home'
)

cfg = dict(
    scene=scene,
    robots={"ur5": {
            cfg_keys.RESOURCE: 'ur5e',
            cfg_keys.ROBOT_MOUNT: 'rethink_stationary',
            cfg_keys.ROBOT_ATTACHMENTS: 'adhesive_gripper'
        }
    },
    task=NullTask,
)

env = gymjoco.from_cfg(cfg=cfg, render_mode='human', frame_skip=5)

N_EPISODES = 1
N_STEPS = 2000

try:
    # run episodes
    for _ in range(N_EPISODES):
        obs, info = env.reset()
        env.render()
        done = False
        i = 0

        pbar = tqdm(total=N_STEPS, desc='running episode')
        while not done and i < N_STEPS:
            i += 1
            action = env.action_space.sample()
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            env.render()
            pbar.update()
        pbar.close()

    # save frames
    # os.makedirs('vidframes', exist_ok=True)
    # for i, frame in enumerate(tqdm(frames, desc='saving frames')):
    #     from PIL import Image
    #
    #     im = Image.fromarray(frame)
    #     im.save(f'vidframes/frame_{i:04d}.png')

except KeyboardInterrupt:
    pass

env.close()
