# GymJoCo
GymJoCo is a python package for creating gymnasium environments using the MuJoCo with easy customization of agents.

## Description
GymJoCo provides a modular framework for creating physics-based simulations and robotic control environments using the MuJoCo physics engine. It's designed with a user-friendly API that allows for easy customization of environments, agents, and tasks.

## Features
- Built on MuJoCo and Gymnasium for reliable physics simulation
- Predefined milestone environments from simple floating objects to complex robotic manipulation tasks
- Customizable robot configurations, including different manipulators and end effectors
- Support for scripted agents, reinforcement learning, and more
- Easy integration with popular RL libraries like Stable Baselines3

## Installation

### Prerequisites
- Python 3.9+ recommended

### Method 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/your-username/GymJoCo.git
cd GymJoCo

# Create and activate conda environment
conda env create -f environment.yml
conda activate gymjoco-env

# Install the package in development mode
pip install -e .
```

### Method 2: Using pip
```bash
# Clone the repository
git clone https://github.com/your-username/GymJoCo.git
cd GymJoCo

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quickstart Guide

### Running a Pre-defined Environment

```python
import gymjoco

# Create a simple environment with rendering
env = gymjoco.make('Milestone-2', render_mode='human')
obs, info = env.reset()

# Run a simple random agent
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Creating a Custom Environment

```python
import gymjoco
from gymjoco.common.defs import cfg_keys
from gymjoco.episode import *
from gymjoco.tasks.null_task import NullTask

# Define scene with objects
scene = SceneSpec(
    '3tableworld',
     objects=(
         ObjectSpec('cubeA', base_pos=[0.14, 0.5, 0.705], base_joints=JointSpec('free')),
         ObjectSpec('cubeB', base_pos=[-0.14, 0.5, 0.705], base_joints=JointSpec('free')),
     ),
    render_camera='rightangleview',
    init_keyframe='home'
)

# Configure the environment
cfg = dict(
    scene=scene,
    robot={
        cfg_keys.RESOURCE: 'ur5e',
        cfg_keys.ROBOT_MOUNT: 'rethink_stationary',
        cfg_keys.ROBOT_ATTACHMENTS: 'adhesive_gripper'
    },
    task=NullTask,
)

# Create environment from configuration
env = gymjoco.from_cfg(cfg=cfg, render_mode='human', frame_skip=5)

# Use the environment
obs, info = env.reset()
# ...
```

### Using with Reinforcement Learning

GymJoCo integrates well with Stable Baselines3 for reinforcement learning:

```python
from stable_baselines3 import PPO
import gymjoco

# Register environments by importing gymjoco
env_id = 'Milestone-1-RL'  # Choose an appropriate environment

# Create and train an RL agent
model = PPO('MlpPolicy', env_id, verbose=1)
model.learn(total_timesteps=10000)

# Test the trained agent
env = gymjoco.make(env_id, render_mode='human')
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()
```

## Available Environments

GymJoCo provides several pre-defined milestone environments:

- `Milestone-1`: Floating ball (translation only)
- `Milestone-1.5`: Floating brick (translation and rotation)
- `Milestone-2`: Paddle pushes ball
- `Milestone-3`: Robot arm pushes object on table
- `Milestone-4`: Robot arm picks up and places a single object
- `Milestone-5`: Robot arm manipulates objects in a cluttered environment
- `Milestone-6`: Robot arm picks up and places multiple objects
- `Milestone-7`: Mobile robot (Fetch) that picks up and places multiple objects

## Examples

Check out the `examples/` directory for more detailed usage examples:

- `scripted_example.py`: Using scripted agents
- `rl_example.py`: Training with reinforcement learning
- `blocksworld.py`: Creating custom environments
- `groceries_example.py`: More complex manipulation tasks


