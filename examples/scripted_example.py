# =============================================================================
# ========================= Scripted Agent Example ============================
# =============================================================================
# This example shows how to create an agent that follows a script based on the current timestep. This is useful for
# creating a baseline for comparison with RL agents, or for creating a simple agent that can be used to test the
# environment. We show an example of a scripted paddle agent that starts at position (-1, 0, 0) and must push a ball
# that starts at position (0, 0, 0) to the desired position (1, 0, 0). The goal is to have the ball in the desired
# when the time runs out. The reward received is the difference between the previous and current distances between the
# ball and the goal. In this example, the agent does so by whacking the ball such that it will be in the right position
# at the final timestep.

# === step 0: import dependencies ===

import numpy as np

import gymjoco
from agents import ScriptedAgent

# === step 1: create environment ===
# create the environment on which to run. to initialize the observation and action spaces, reset the environment. to
# enable rendering, set the argument `render_mode='human'`.

# env = gymjoco.make('Paddleworld-v0')  # without rendering
env = gymjoco.make('Milestone-2', render_mode='human')  # with rendering
env.reset()


# === step 2: create agent script ===
# create a script for the agent to follow. this is a function that accepts the current timestep and returns the action
# to perform at that state.

def script(timestep):
    # # random action script
    # return agent.env.action_space.sample()

    # script for paddleworld move ball from 0,0 to 1, 0
    action = np.zeros(agent.env.action_space.shape)  # get zero action of correct shape
    action[0] = (3.4 if timestep < 500  # move paddle toward the ball
                 else -5 if timestep < 700  # break paddle after whack
                 else 0)  # idle paddle while the ball floats toward the goal
    return action


# === step 3: create agent ===
# use the `ScriptedAgent` helper class to create an agent that follows the script. this agent uses the stable-baselines3
# API and sis somewhat compatible with it, provided a step counting callback is used.

agent = ScriptedAgent(script, env=env)

# === step 4: run agent ===
# run the agent for a single episode. For multiple episodes, use the `n_episodes` argument. the agent will automatically
# keep track of the timesteps and reset the script when the episode ends. The returned values are the accumulated reward
# and the length of each episode run. This function calls the `evaluate_policy` function from stable-baselines3 using
# an internal step counting callback. However, you can still add another callback as an argument which will be called
# after the step counting callback. One use case for this is to save additional metrics, e.g., success rate.

ep_r, ep_l = agent.run()
print(f'Average reward: {np.mean(ep_r):.2f} +/- {np.std(ep_r):.2f}')
