from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco

import time

class Unimal(PipelineEnv):

  def __init__(
      self,
      xml_path, 
      ctrl_cost_weight=0.5,
      use_contact_forces=False,
      contact_cost_weight=5e-4,
      healthy_reward=1.0,
      terminate_when_unhealthy=True,
      healthy_z_range=(0.2, 1.0),
      contact_force_range=(-1.0, 1.0),
      reset_noise_scale=0.1,
      exclude_current_positions_from_observation=True,
      **kwargs,
  ):

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 1
    mj_model.opt.ls_iterations = 1

    sys = mjcf.load_model(mj_model)

    physics_steps_per_control_step = 5
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

    # sys = mjcf.load(xml_path)

    # n_frames = 5

    # if backend in ['spring', 'positional']:
    #   sys = sys.replace(dt=0.005)
    #   n_frames = 10

    # if backend == 'positional':
    #   # TODO: does the same actuator strength work as in spring
    #   sys = sys.replace(
    #       actuator=sys.actuator.replace(
    #           gear=200 * jp.ones_like(sys.actuator.gear)
    #       )
    #   )

    # kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

    # super().__init__(sys=sys, backend=backend, **kwargs)

    self._ctrl_cost_weight = ctrl_cost_weight
    self._use_contact_forces = use_contact_forces
    self._contact_cost_weight = contact_cost_weight
    self._healthy_reward = healthy_reward
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._contact_force_range = contact_force_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = (
        exclude_current_positions_from_observation
    )

    # self.get_limb_context()
    # self.get_joint_context()
    self.get_action_index()

    if self._use_contact_forces:
      raise NotImplementedError('use_contact_forces not implemented.')
    
    print ('successfully initialize!')

  def get_limb_context(self):
    # get context features
    # limb hardware features
    # body_pos: transform for the link frame relative to the parent frame
    # TODO: the z axis of torso in body_pos is not consistent with Mojoco
    body_pos = self.sys.link.transform.pos.copy()
    # body_ipos: position of the inertial frame relative to the link frame
    body_ipos = self.sys.link.inertia.transform.pos.copy()
    # body_iquat: rotation of the inertial frame relative to the link frame
    body_iquat = self.sys.link.inertia.transform.rot.copy()
    # geom_quat: 
    geom_quat = jp.concatenate([geom.transform.rot.copy() for geom in self.sys.geoms], axis=0)
    # body mass
    # TODO: the mass is slightly larger than that in Mujoco. Why?
    body_mass = self.sys.link.inertia.mass.copy().reshape(-1, 1)
    # body shape
    body_shape = []
    for geom in self.sys.geoms:
      if type(geom) == base.Sphere:
        body_shape.append(jp.append(geom.radius.copy(), 0.).reshape(1, -1))
      elif type(geom) == base.Capsule:
        # the length is twice as in Mujoco, which should be fine
        body_shape.append(jp.stack([geom.radius.copy(), geom.length.copy()], axis=1))
    body_shape = jp.concatenate(body_shape)
    self.limb_context = jp.concatenate([body_pos, body_ipos, body_iquat, geom_quat, body_mass, body_shape], axis=1)

  def get_joint_context(self):
    # joint hardware features
    # TODO: joint_pos
    # joint_pos = self.sys.dof.joint_pos
    # joint_range
    joint_pos = jp.stack([self.sys.dof.limit[0][6:].copy(), self.sys.dof.limit[1][6:].copy()], axis=1)
    # TODO: joint_axis
    # joint_axis = self.sys.dof
    # gear
    gear = self.sys.actuator.gear.copy().reshape(-1, 1)
    self.joint_context = jp.concatenate([joint_pos, gear], axis=1)

  def get_action_index(self):
    # mask the joints for each limb
    self.limb_num = self.sys.num_links()
    dof_link_idx = self.sys.dof_link()[6:].copy()
    repeat_mask = (dof_link_idx[1:] == dof_link_idx[:-1])
    repeat_mask = jp.insert(repeat_mask, 0, 0)
    self.action_index = dof_link_idx * 2 + repeat_mask

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

    data = self.pipeline_init(qpos, qvel)
    obs = self._get_obs(data)
    print (obs.shape)

    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_forward': zero,
        'reward_survive': zero,
        'reward_ctrl': zero,
        'x_position': zero,
        'y_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
        'forward_reward': zero,
    }
    return State(data, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Run one timestep of the environment's dynamics."""

    # remove useless action dimensions
    action = action[self.action_index]

    # step
    pipeline_state0 = state.pipeline_state
    pipeline_state = self.pipeline_step(pipeline_state0, action)

    # forward reward
    velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
    forward_reward = velocity[0]

    # height check and healthy reward
    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, x=0.0, y=1.0)
    is_healthy = jp.where(
        pipeline_state.x.pos[0, 2] > max_z, x=0.0, y=is_healthy
    )
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    # control cost
    ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

    obs = self._get_obs(pipeline_state)
    reward = forward_reward + healthy_reward - ctrl_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    state.metrics.update(
        reward_forward=forward_reward,
        reward_survive=healthy_reward,
        reward_ctrl=-ctrl_cost,
        x_position=pipeline_state.x.pos[0, 0],
        y_position=pipeline_state.x.pos[0, 1],
        distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        forward_reward=forward_reward,
    )

    return state.replace(
        pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
    )

  def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:

    # limb proprioceptive features
    xpos = pipeline_state.x.pos.copy()
    xquat = pipeline_state.x.rot.copy()
    xvelp = pipeline_state.xd.vel.copy()
    xvelr = pipeline_state.xd.ang.copy()
    limb_proprioceptive = jp.concatenate([xpos, xquat, xvelp, xvelr], axis=1)

    # joint proprioceptive features
    # TODO: not sure if this split is correct
    # TODO: normalize qpos with joint_angle
    qpos = pipeline_state.q[7:].copy().reshape(-1, 1)
    qvel = pipeline_state.qd[6:].copy().reshape(-1, 1)
    joint_proprioceptive = jp.concatenate([qpos, qvel], axis=1)

    # merge limb and joint features together
    limb_obs, joint_obs = limb_proprioceptive, joint_proprioceptive
    # limb_obs = jp.concatenate([limb_proprioceptive, self.limb_context], axis=1)
    # joint_obs = jp.concatenate([joint_proprioceptive, self.joint_context], axis=1)
    # mask the joints for each limb
    joint_obs_padded = jp.zeros([self.limb_num * 2, joint_obs.shape[1]])
    joint_obs_padded.at[self.action_index, :].set(joint_obs)
    joint_obs_padded = joint_obs_padded.reshape(self.limb_num, -1)

    obs = jp.concatenate([limb_obs, joint_obs_padded], axis=1)

    return obs.ravel()
