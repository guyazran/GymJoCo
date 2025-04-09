from typing import Optional, Dict

from dm_control import mjcf
import mujoco

from .entity import Entity
from .mjcf_composer import MJCFComposer
from .mjcf_utils import physics_from_mjcf_model
from .robot_agent import RobotAgent
from ..common.defs.types import InfoDict, Vector
from ..episode.specs.robot_spec import RobotSpec
from ..episode.specs.scene_spec import SceneSpec

import tempfile


class Simulator:
    """
    An interface for initializing and interacting with MuJoCo simulations.

    A `Simulator` instance uses a specifications for a scene and a robots to compose a model for the simulation.
    Specifications are defined according to the spec dataclasses (see gymjoco.episode.specs). A specification points
    to MJCF (XML) asset files that are loaded and merged according to the specification details.

    public fields:
        - scene / robots:
            The scene and robots specification instances set at instance construction or using the `swap_specs` method.

        - composer:
            An MJCF composition tool for merging episode-specific MJCF files.

        - physics
            A representation of the simulation physics (see dm_control.mjcf.Physics). This is a wrapper around the
            MuJoCo model and data objects.

        - mjcf_model:
            An access variable to the MJCF model object on which the simulation model is based.

        - model:
            An access variable to the pointer of the MuJoCo model object in the simulation physics.

        - data:
            An access variable to the pointer of the MuJoCo data object in the simulation physics.

    public methods:
        - initialize:
            Composes and initializes the simulation model.

        - step:
            Steps the simulation a specified number of time ticks.

        - reset:
            Resets the simulation to the initial state.

        - swap_specs:
            Swaps the scene and robot specifications of the simulation.

        - get_agent:
            Creates new RobotAgent instances for the robots in the simulation.

        - get_privileged_info:
            Constructs an information dictionary containing privileged information.

        - free:
            Frees the simulation physics memory.

        - get_entity:
            Get an `Entity` object bound to an element in the simulation.
    """

    def __init__(self, scene: SceneSpec, robots: Dict[str, RobotSpec]) -> None:
        """
        Creates a new MuJoCo simulation according to the given scene and robot specifications.
        :param scene: the scene specification.
        :param robots: a dictionary of robot specifications.
        """

        # set scene and robot specifications
        self.scene = scene
        self.robots = robots

        # declare MJCF composition tool
        self.composer = MJCFComposer()

        # declare simulation keyframes
        self.keyframes: dict[str, tuple[mjcf.RootElement, Vector, Vector, Vector, Vector]] = {}

        # initialize simulation fields
        self.physics: Optional[mjcf.Physics] = None

        # initialize the simulation model and data
        self.initialize()

    def __del__(self) -> None:
        """Frees the simulation physics memory when the instance is deleted."""
        self.free()

    # ========================= #
    # ========== API ========== #
    # ========================= #

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        """An access variable to the MJCF model object on which the simulation model is based."""
        return self.composer.mjcf_model

    @property
    def model(self) -> mujoco.MjModel:
        """An access variable to the pointer of the MuJoCo model object in the simulation physics."""
        return self.physics.model.ptr

    @property
    def data(self) -> mujoco.MjData:
        """An access variable to the pointer of the MuJoCo data object in the simulation physics."""
        return self.physics.data.ptr

    def initialize(self) -> None:
        """Composes and initializes the simulation model."""
        # create mjcf for model
        self.__compose_mjcf()

        # create mujoco simulation
        self.free()  # free previous simulation
        self.physics = physics_from_mjcf_model(self.mjcf_model)

    def step(self, n_frames) -> None:
        """
        Step the simulator a specified number of time ticks
        :param n_frames: The number of simulated time ticks to step
        """
        self.physics.step(nstep=n_frames)

        # As of MuJoCo 2.0, force-related quantities like cacc are not computed
        # unless there's a force sensor in the model.
        # See https://github.com/openai/gym/issues/1541
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def reset(self) -> None:
        """
        Resets the simulation to the initial state. The initial state is determined in the following process:
            1. Set all state variables to their initial state as defined in the model.
            2. Update state variables according to the scene's `init_keyframe` filed (if specified).
            3. Override the robot and object state variables with specified `init_qpos/qvel/act` fields.
        """
        # reset simulation data
        self.physics.reset()

        # set initial state according to selected keyframe
        self.__set_keyframe_state()

        # set initial state for individual entities in the model (overrides keyframe)
        self.__set_init_state()

    def swap_specs(self, scene: SceneSpec, robots: Dict[str, RobotSpec]) -> None:
        """
        Swaps the scene and robot specifications of the simulation to accommodate multiple robots.
        This method is used to change the scene and robots such that the simulation is reinitialized
        only when it is absolutely necessary.
        :param scene: the new scene specification.
        :param robots: a dictionary of new robot specifications.
        """
        # swap scene and robot
        self.scene, scene = scene, self.scene
        self.robots, robots = robots, self.robots

        # check if new model required
        if (SceneSpec.require_different_models(self.scene, scene) or
                any(RobotSpec.require_different_models(self.robots[name], robots[name])
                    for name in self.robots.keys() & robots.keys())):
            self.initialize()
        else:
            # swap composer spec keys
            # it is enough to swap only the scene objects and the robot (+attachments) since they make up all the
            # keys in the internal composer dictionary
            self.composer.swap_spec_ids(self.scene.objects, scene.objects, self.robots.values(), robots.values())

    def get_agents(self) -> Dict[str, RobotAgent]:
        """Create and return RobotAgent instances for each robot."""
        return {name: RobotAgent(name, self) for name, robot in self.robots.items()}

    def get_privileged_info(self) -> InfoDict:
        """
        Constructs an information dictionary containing privileged information that is not available outside of
        simulation.
        :return: A dictionary containing the MuJoCo model and data objects of the current running simulation.
        """
        return dict(
            model=self.model,
            data=self.data
        )

    def free(self) -> None:
        """Frees the simulation physics memory."""
        if self.physics is not None:
            self.physics.free()
            self.physics = None

    def get_entity(self, name: str, tag: str = 'body') -> Entity:
        """
        Get an `Entity` object (see `gymjoco.simulation.entity.Entity`) bound to an element corresponding to the
        specified name and tag.
        :param name: the name attribute of the element to bind.
        :param tag: the XML tag of the element to bind.
        :return: An `Entity` object that binds the retrieved element.
        :raises: ValueError if the element is not found.
        """
        return Entity.from_name_and_tag(name, tag, self.mjcf_model, self.physics)

    # ================================== #
    # ========== init helpers ========== #
    # ================================== #

    def __compose_mjcf(self):
        self.composer.set_base_scene(self.scene)
        for robot_name, robot_spec in self.robots.items():
            self.composer.attach_robot(robot_spec)
        self.keyframes = self.composer.extract_keyframes()

    def __set_keyframe_state(self):
        # get keyframe ID
        keyframe = self.scene.init_keyframe

        if keyframe is None:
            return  # no keyframe provided

        try:
            if isinstance(keyframe, str):
                keyframe_data = self.keyframes[keyframe]
            else:
                keyframe_data = list(self.keyframes.values())[keyframe]
        except (KeyError, IndexError):
            raise ValueError(f'Invalid keyframe ID: {keyframe}')

        for root, qpos, qvel, act, ctrl in keyframe_data:
            root_entity = Entity.from_model(root, self.physics)
            root_entity.set_state(qpos, qvel, act, ctrl, recursive=False)

    def __set_init_state(self):
        for obj in self.scene.objects:
            addon_body = Entity(self.composer.get_object(obj), self.physics)
            addon_body.configure_joints(obj.init_pos, obj.init_vel)

        for agent in self.get_agents().values():
            agent.reset()
