from . import milestones  # registers the milestones
from gymnasium import make
from .common import cfg_keys
from .episode import *
from .env import GymJoCoEnv
from .tasks.rearrangement.rearrangement_task import COMRearrangementTask as _COMTask
from ._version import __version__

__all__ = [
    'make',
    'GymJoCoEnv',
    'from_cfg',
    'from_cfg_file',
    'cfg_keys',
    'EpisodeSpec',
    'SceneSpec',
    'RobotSpec',
    'TaskSpec',
    'ObjectSpec',
    'AttachmentSpec',
    'MountSpec'
]

from_cfg = GymJoCoEnv.from_cfg
from_cfg_file = GymJoCoEnv.from_cfg_file
