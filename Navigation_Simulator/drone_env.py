"""
Inpired by humanoid_v4.py in Gymnasium mujoco examples

Requires the gymnasium/envs/mujoco/mujoco_env.py to be replaced by the modified version in this repository
"""
import os
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from simple_pid import PID
from scipy.spatial.transform import Rotation
import mujoco
from path_planning import spline_path

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

def _wrap_angle_error(angle_error):
  if angle_error > np.pi:
    return angle_error - 2 * np.pi
  elif angle_error < -np.pi:
    return angle_error + 2 * np.pi
  else:
    return angle_error

def sqrt_controller(Kp, error):
      if abs(error) < 30/180 * np.pi:
        # Square root region
        return np.sign(error) * np.sqrt(Kp*abs(error))
      else:
        return Kp*error

class DroneEnv(MujocoEnv, utils.EzPickle):
    """
    TODO Insert desription here later
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(
        self,
        forward_reward_weight=1.25,
        # ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.1, 3.0),
        reset_noise_scale=1e-2,
        n_gates=5,
        **kwargs,
    ):
        # utils.EzPickle.__init__(
        #     self,
        #     forward_reward_weight,
        #     ctrl_cost_weight,
        #     healthy_reward,
        #     terminate_when_unhealthy,
        #     healthy_z_range,
        #     reset_noise_scale,
        #     **kwargs,
        # )

        self._forward_reward_weight = forward_reward_weight
        # self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            os.path.join(os.path.dirname(__file__),"mujoco_menagerie-main/skydio_x2_racecourse/scene.xml"),
            1,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.mass = 1.325 # change this if using different model

        # Rate controller PIDs
        self.pid_thrust = PID(5, 0.5, 0.01, setpoint=0)
        self.pid_roll_rate = PID(1, 0.5, 0.01, setpoint=0, output_limits= (-1,1))
        self.pid_pitch_rate = PID(1, 0.5, 0.01, setpoint=0, output_limits= (-1,1))
        self.pid_yaw_rate = PID(5, 0.5, 0.01, setpoint=0, output_limits= (-1,1))

        # Attitude controller PIDs
        self.pid_alt = PID(1, 0, 0, setpoint=0, output_limits= (-1,1))
        self.Kp = 2
        self.Kp_yaw = 1
        # Unused - using SQRT controller instead
        # self.pid_roll = PID(3, 2, 5, setpoint=0, output_limits= (-1,1))
        # self.pid_pitch = PID(3, 2, 5, setpoint=0, output_limits= (-1,1))
        # self.pid_yaw = PID(5, 1, 5, setpoint=0, output_limits= (-1,1))

        # Velocity XY controller PIDs
        self.pid_vx = PID(10, 0.2, 3, setpoint=0, output_limits= (-10,10))
        self.pid_vy = PID(10, 0.2, 3, setpoint=0, output_limits= (-10,10))

        self.Kp_pos = 1
        # Gate positions
        mujoco.mj_forward(self.model, self.data)
        self.n_gates = n_gates
        self.gate_centres = []
        self.control_points = [[self.data.qpos[0],self.data.qpos[1],self.data.qpos[2]]]
        self.gate_d = 0.5

        for i in range(self.n_gates):
            pos = self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gate"+str(i+1))]   # numpy array [x, y, z]
            pos[2] += 0.8 # gate position in xml file corresponds to base, shift up 0.8 for gate centre
            print(f"Gate {i+1} position: {pos}")

            # Also add waypoints in front and behind of gate
            quat = self.data.xquat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gate"+str(i+1))]
            euler_gate = Rotation.from_quat(quat).as_euler('zyx',degrees=False)
            # print(euler_gate)

            yaw = euler_gate[2]
            pos1 = pos - self.gate_d * np.array([np.sin(yaw), np.cos(yaw), 0])
            pos2 = pos + self.gate_d * np.array([np.sin(yaw), np.cos(yaw), 0])
            self.control_points.append(pos1)
            self.gate_centres.append(pos)
            self.control_points.append(pos)
            self.control_points.append(pos2)

        # For spline calculation
        self.control_points.append([0,0,0])

        points_x = []
        points_y = []
        points_z = []
        for i in range(len(self.control_points)):
            points_x.append(self.control_points[i][0])
            points_y.append(self.control_points[i][1])
            points_z.append(self.control_points[i][2])
        
        # Compute path
        self.spline_x, self.spline_y, self.spline_z, self.spline_vx, self.spline_vy, self.spline_vz = spline_path(points_x,points_y,points_z)
        self.spline_idx = 0

    # Overriding in BaseMujocoEnv
    def _set_action_space(self):
        self.action_space = Box(low=np.array([-2, -2, -2]), high=np.array([2, 2, 2]), dtype=np.float32)
        return self.action_space
    
    def get_position(self):
        return self.data.qpos # (x y z q1 q2 q3 q4)

    def get_velocities(self):
        return self.data.qvel # (vx vy vz wx wy wz)

    # def reset_sim(self):
    #     # Set drone to hover
    #     mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def _rates_to_motor(thrust, roll_rate, pitch_rate, yaw_rate):
        return np.array([
        thrust + roll_rate + pitch_rate - yaw_rate,
        thrust - roll_rate + pitch_rate + yaw_rate,
        thrust - roll_rate - pitch_rate - yaw_rate,
        thrust + roll_rate - pitch_rate + yaw_rate
        ])

    def ctrl_att_rate(self, target_vz, target_roll_rate, target_pitch_rate, target_yaw_rate):

        self.pid_thrust.setpoint = target_vz
        self.pid_roll_rate.setpoint = target_roll_rate
        self.pid_pitch_rate.setpoint = target_pitch_rate
        self.pid_yaw_rate.setpoint = target_yaw_rate

        # self.target_roll_rate_hist.append(target_roll_rate)
        # self.target_pitch_rate_hist.append(target_pitch_rate)
        # self.target_yaw_rate_hist.append(target_yaw_rate)

        return DroneEnv._rates_to_motor(self.pid_thrust(self.vz) + 3.249,
                                            self.pid_roll_rate(self.roll_rate),
                                            self.pid_pitch_rate(self.pitch_rate),
                                            self.pid_yaw_rate(self.yaw_rate))

    def ctrl_att_hold(self, target_roll, target_pitch, target_yaw, target_vz=0):

        # Store target attitudes
        # self.target_roll_hist.append(target_roll)
        # self.target_pitch_hist.append(target_pitch)
        # self.target_yaw_hist.append(target_yaw)
        
        # self.pid_alt.setpoint = target_alt
        # self.pid_roll.setpoint = target_roll
        # self.pid_pitch.setpoint = target_pitch
        # self.pid_yaw.setpoint = target_yaw

        r2 = Rotation.from_euler('zyx', np.array([target_roll, target_pitch, target_yaw]), degrees=False)

        rot = r2 * self.r1.inv() # to get relative rotation between attitude and target in the world frame
        euler_error = rot.as_euler('zyx', degrees=False)
        roll_error = _wrap_angle_error(euler_error[0])
        pitch_error = _wrap_angle_error(euler_error[1])
        yaw_error = _wrap_angle_error(euler_error[2])
        # Store attitude errors
        # self.roll_error_hist.append(roll_error)
        # self.pitch_error_hist.append(pitch_error)
        # self.yaw_error_hist.append(yaw_error)

        # P control
        # target_rise_rate = self.pid_alt(self.altitude)
        # target_roll_rate_world = self.pid_roll(self.roll)
        # target_pitch_rate_world = self.pid_pitch(self.pitch)
        # target_yaw_rate_world = self.pid_yaw(self.yaw)
        
        target_roll_rate_world = -sqrt_controller(self.Kp, roll_error)
        target_pitch_rate_world = -sqrt_controller(self.Kp, pitch_error)
        target_yaw_rate_world = sqrt_controller(self.Kp_yaw, yaw_error)

        # Transform to body frame
        target_p= target_roll_rate_world - np.sin(self.pitch) * target_yaw_rate_world
        target_q = np.cos(self.roll) * target_pitch_rate_world + np.sin(self.roll) * np.cos(self.pitch) * target_yaw_rate_world
        target_r = -np.sin(self.roll) * target_pitch_rate_world + np.cos(self.pitch) * np.cos(self.roll) * target_yaw_rate_world

        # self.ctrl_att_rate(target_rise_rate, target_p, target_q, target_r)
        return self.ctrl_att_rate(target_vz, target_roll_rate_world, target_pitch_rate_world, target_yaw_rate_world)
    
    def ctrl_vel(self, target_vx, target_vy, target_vz):
        g = 9.8
        vx_error = target_vx - self.vx
        vy_error = target_vy - self.vy

        # self.vx_error_hist.append(vx_error)
        # self.vy_error_hist.append(vy_error)

        # Kp_vel = 5
        # target_pitch = Kp_vel * np.arctan(vx_error/g)
        # target_roll = Kp_vel * np.arctan(np.cos(target_pitch)*vy_error/g)

        self.pid_vx.setpoint = target_vx
        self.pid_vy.setpoint = target_vy
        target_ax = self.pid_vx(self.vx)
        target_ay = self.pid_vy(self.vy)
        target_pitch = np.arctan(target_ax/g)
        target_roll = np.arctan(np.cos(target_pitch)*target_ay/g)

        # print(f'Attitude hold {target_roll}, {target_pitch}, 0, {target_vz}')
        return self.ctrl_att_hold( target_roll, target_pitch, np.pi, target_vz=target_vz)
    
    def ctrl_pos_hold(self, x, y, z, ff_vx=0, ff_vy=0, ff_vz=0):
        x_error = self.x - x
        y_error = self.y - y
        z_error = self.z - z

        # self.target_x_hist.append(x)
        # self.target_y_hist.append(y)
        # self.target_z_hist.append(z)

        target_vx = -sqrt_controller(self.Kp_pos, x_error)
        target_vy = -sqrt_controller(self.Kp_pos, y_error)
        target_vz = -sqrt_controller(self.Kp_pos, z_error)
        # print(f'Target v: {[target_vx, target_vy, target_vz]}')
        # print(f'Feedforward v: {[ff_vx, ff_vy, ff_vz]}')

        return self.ctrl_vel(target_vx + ff_vx, target_vy + ff_vy, target_vz+ff_vz)
    
    def follow_spline(self,spline_step=1, wp_error = 0.1):
        # Increment spline index if close enough to next waypoint
        idx = self.spline_idx%len(self.spline_x)
        next_waypoint = [self.spline_x[idx], self.spline_y[idx], self.spline_z[idx]]
        pos_error = np.sqrt( (self.x-next_waypoint[0])**2 + (self.y-next_waypoint[1])**2 + (self.z-next_waypoint[2])**2)
        if pos_error < wp_error:
            self.spline_idx += spline_step
            # print(f'Idx = {self.spline_idx}')

        # Set position hold to waypoint with feedforward velocity scaled by position error
        idx = self.spline_idx%len(self.spline_x)
        next_waypoint = [self.spline_x[idx], self.spline_y[idx], self.spline_z[idx]]
        pos_error = np.sqrt( (self.x-next_waypoint[0])**2 + (self.y-next_waypoint[1])**2 + (self.z-next_waypoint[2])**2)
        K_ff = 5 * np.min([wp_error/pos_error, 1])
        
        return self.ctrl_pos_hold(next_waypoint[0], next_waypoint[1], next_waypoint[2], ff_vx=K_ff * self.spline_vx[idx], ff_vy=K_ff * self.spline_vy[idx], ff_vz=K_ff * self.spline_vz[idx])

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    # def control_cost(self, action):
    #     control_cost = self._ctrl_cost_weight * np.sum(np.square(self.data.ctrl))
    #     return control_cost

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        # print(f'{self.data.qpos[2]} is healthy: {is_healthy}')

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if self._terminate_when_unhealthy else False
        return terminated
    
    def _update_state(self):
        # Read pose
        pos = self.get_position()
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        q1 = pos[3]
        q2 = pos[4]
        q3 = pos[5]
        q4 = pos[6]
        self.r1 = Rotation.from_quat(np.array([q1,q2,q3,q4]))
        euler_angle = self.r1.as_euler('zyx',degrees=False)
        # print(euler_angle)
        self.roll = euler_angle[0]
        self.pitch = euler_angle[1]
        self.yaw = euler_angle[2]

        # Read rates
        vel = self.get_velocities()
        self.vx = vel[0]
        self.vy = vel[1]
        self.vz = vel[2]
        self.roll_rate = -vel[3]
        self.pitch_rate = vel[4]
        self.yaw_rate = -vel[5]

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        # com_inertia = self.data.cinert.flat.copy()
        # com_velocity = self.data.cvel.flat.copy()

        # actuator_forces = self.data.qfrc_actuator.flat.copy()
        # external_contact_forces = self.data.cfrc_ext.flat.copy()

        # print(f'position = {position}')
        # print(f'velocity = {velocity}')
        # print(f'com_inertia = {com_inertia}')
        # print(f'com_velocity = {com_velocity}')
        # print(f'actuator_forces = {actuator_forces}')
        # print(f'external_contact_forces = {external_contact_forces}')

        return np.concatenate(
            (
                position,
                velocity
                # com_inertia,
                # com_velocity,
                # actuator_forces,
                # external_contact_forces,
            )
        )
    
    def _calculate_reward(self):

        # ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * self.vx
        healthy_reward = self.healthy_reward
        stability_penalty = -0.01 * (abs(self.roll) + abs(self.pitch))

        rewards = forward_reward + healthy_reward + stability_penalty
        # reward = rewards - ctrl_cost

        return forward_reward

    def step(self, action):

        self._update_state()
        
        # ctrl = self.follow_spline(spline_step=5, wp_error=0.5)
        ctrl = self.ctrl_vel(action[0],action[1],action[2])

        self.do_simulation(ctrl, n_frames=5)

        self._update_state()

        observation = self._get_obs()
        # print(f'obs = {observation}')

        reward = self._calculate_reward()
        print(f'{action} -> {reward}')
        
        terminated = self.terminated
        info = {
            # "reward_linvel": forward_reward,
            # "reward_quadctrl": -ctrl_cost,
            # "reward_alive": healthy_reward,
            "x_position": self.x,
            "y_position": self.y,
            # "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": self.vx,
            "y_velocity": self.vy,
            # "forward_reward": forward_reward,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

