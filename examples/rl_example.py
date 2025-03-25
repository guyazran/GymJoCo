# =============================================================================
# ================================ RL Example =================================
# =============================================================================
# This example trains an RL agent on the gymjoco environments using the stable-baselines3 API.

# === step 0: import dependencies ===

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# NOTE: we must import gymjoco so that the environment will be registered
from examples.utils.callbacks import LogInfoCallback, LogAgentRewardCallback

# === step 1: choose environment and configurations ===

# choose environment
ENV_ID = 'Milestone-1-RL'

# set random seed
SEED = 42

# dump directories
TENSORBOARD_DIR = 'dumps/tensorboard'
MODELS_DIR = 'dumps/models'
CHKP_DIR = 'dumps/checkpoints'

# === step 2: create agent ===
# the gymjoco environments are registered with gymnasium, so we can use the stable-baselines3 API to create an agent
# with just the string ID of the environment. choose any RL algorithm that supports continuous action spaces. Since our
# observations are structured continuous vectors, we use a simple MLP policy with the default configurations. We also
# set the random seed for consistency. Everything is logged to tensorboard for easy visualization of the training
# process.

# model = DDPG(policy='MlpPolicy', env=ENV_ID, tensorboard_log='dumps/tensorboard', seed=SEED)
# model = SAC(policy='MlpPolicy', env=ENV_ID, tensorboard_log='dumps/tensorboard', seed=SEED)
model = PPO(policy='MlpPolicy', env=ENV_ID, tensorboard_log=TENSORBOARD_DIR, seed=SEED)

# === step 3: train agent ===
# train the agent for 1M steps. We save the model every 10k steps and display a progress bar.

try:
    model.learn(
        # run for 1M steps
        total_timesteps=1_000_000,

        # callbacks for logging
        callback=[
            CheckpointCallback(save_freq=10_000, save_path=CHKP_DIR),  # save model every 10k steps
            LogInfoCallback('task_reward', episode_agg=sum),  # log the task (acc) reward (actual reward we care about)
            LogAgentRewardCallback()
        ],  # log reward from the agent's perspective

        # display a progress bar
        progress_bar=True
    )
except KeyboardInterrupt:
    print(' training interrupted by user')

# === step 4: save model ===
# save the model and checkpoints to a directory matching the tensorboard log directory name.

# get the name of the tb log directory
run_name = Path(model.logger.dir).name

# save the model matching the tb log directory name
model.save(Path(MODELS_DIR) / run_name)

# move checkpoints into a directory matching the tb log directory name
print('moving checkpoints')
chkp_out = Path(CHKP_DIR) / run_name
chkp_out.mkdir(parents=True, exist_ok=True)
for chkp in Path(CHKP_DIR).glob('*.zip'):
    chkp.rename(chkp_out / chkp.name)
