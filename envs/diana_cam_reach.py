#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        diana_cam_reach.py
 @Created:          2022/6/20 11:23
 @Software:         PyCharm

 @Author:           HHH
 @Email:            1950049@tongji.edu.cn
 @Gitee:            https://gitee.com/jin-yiyang
 @Version:          v1.0

 @Description:      Main Function:
 ------------------------------------------------------------------
 @Change History :
  <Date>     | <Version> | <Author>       | <Description>
 ------------------------------------------------------------------
  2022/6/20  | v1.0      | HHH            | Create file
 ------------------------------------------------------------------
'''

import pybullet as p
import pybullet_data
import os
import sys
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import sqrt
import random
import time
import math
import cv2
from termcolor import colored
import torch
import matplotlib.pyplot as plt
from colorama import Fore, init, Back
import sys
import os
import inspect

current_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
os.chdir(current_dir)
sys.path.append('../')

init(autoreset=True)  # this lets colorama takes effect only in current line.

class KukaCamReachEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    max_steps_one_episode = 10000

    def __init__(self, is_render=False, is_good_view=False):

        # some camera parameters
        self.camera_parameters = {
            'width': 960.,
            'height': 720,
            'fov': 60,
            'near': 0.1,
            'far': 100.,
            'eye_position': [0.59, 0, 0.8],
            'target_position': [0.55, 0, 0.05],
            'camera_up_vector':
            [1, 0, 0],  # I really do not know the parameter's effect.
            'light_direction': [
                0.5, 0, 1
            ],  #the direction is from the light source position to the origin of the world frame.
        }

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.55, 0, 0.05],
            distance=.7,
            yaw=90,
            pitch=-70,
            roll=0,
            upAxisIndex=2)

        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_parameters['fov'],
            aspect=self.camera_parameters['width'] /
            self.camera_parameters['height'],
            nearVal=self.camera_parameters['near'],
            farVal=self.camera_parameters['far'])

        self.is_render = is_render
        self.is_good_view = is_good_view

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        x_off=0.1
        self.x_low_obs = 0.2+x_off
        self.x_high_obs = 0.7+x_off
        self.y_low_obs = -0.3
        self.y_high_obs = 0.3
        self.z_low_obs = 0
        self.z_high_obs = 0.55

        self.x_low_action = -0.4
        self.x_high_action = 0.4
        self.y_low_action = -0.4
        self.y_high_action = 0.4
        self.z_low_action = -0.6
        self.z_high_action = 0.3

        p.setAdditionalSearchPath('../models/')
        p.configureDebugVisualizer(lightPosition=[5, 0, 5])
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(low=np.array(
            [self.x_low_action, self.y_low_action, self.z_low_action]),
                                       high=np.array([
                                           self.x_high_action,
                                           self.y_high_action,
                                           self.z_high_action
                                       ]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 84, 84))

        self.step_counter = 0

        self.urdf_root_path = pybullet_data.getDataPath()
        # lower limits for null space
        self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
        # upper limits for null space
        self.upper_limits = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # joint ranges for null space
        self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # restposes for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        self.init_joint_positions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
            -0.006539
        ]

        self.orientation = p.getQuaternionFromEuler(
            [0., 0.,0.])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_counter = 0

        p.resetSimulation()
        self.terminated = False
        p.setGravity(0, 0, -10)

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])
        self.kuka_id = p.loadURDF("diana/DianaS1_robot.urdf",
                                  baseOrientation=p.getQuaternionFromEuler(
                                      [0., 0., math.pi]),
                                  useFixedBase=True)
        table_uid = p.loadURDF(os.path.join(self.urdf_root_path,
                                            "table/table.urdf"),
                               basePosition=[0.5, 0, -0.65])
        p.changeVisualShape(table_uid, -1, rgbaColor=[1, 1, 1, 1])

        self.object_id = p.loadURDF(os.path.join(self.urdf_root_path,
                                                 "random_urdfs/000/000.urdf"),
                                    basePosition=[
                                        random.uniform(self.x_low_obs,
                                                       self.x_high_obs),
                                        random.uniform(self.y_low_obs,
                                                       self.y_high_obs), 0.01
                                    ])

        self.num_joints = p.getNumJoints(self.kuka_id)

        for i in range(1, self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i - 1],
            )

        for i in range(self.num_joints):
             print(p.getJointInfo(self.kuka_id, i))

        self.robot_pos_obs = p.getLinkState(self.kuka_id,
                                            self.num_joints - 1)[4]
        p.stepSimulation()

        (_, _, px, _,
         _) = p.getCameraImage(width=960,
                               height=960,
                               viewMatrix=self.view_matrix,
                               projectionMatrix=self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        self.images = px

        p.enableJointForceTorqueSensor(bodyUniqueId=self.kuka_id,
                                       jointIndex=self.num_joints - 1,
                                       enableSensor=True)

        # print(Fore.GREEN+'force_sensor={}'.format(self._get_force_sensor_value()))
        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        self.images = self.images[:, :, :3]  # the 4th channel is alpha channel, we do not need it.

        return self._process_image(self.images)

    def _process_image(self, image):
        """Convert the RGB pic to gray pic and add a channel 1
        Args:
            image ([type]): [description]
        """
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(image, (84, 84))[None, :, :] / 255.
            return image
        else:
            return np.zeros((1, 84, 84))

    def step(self, action):
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.new_robot_pos = [
            self.current_pos[0] + dx, self.current_pos[1] + dy,
            self.current_pos[2] + dz
        ]
        self.robot_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=[
                self.new_robot_pos[0], self.new_robot_pos[1],
                self.new_robot_pos[2]
            ],
            targetOrientation=self.orientation,
            jointDamping=self.joint_damping,
        )
        for i in range(1, self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i - 1],
            )
        p.stepSimulation()

        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1

        return self._reward()

    def _reward(self):

        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.object_state = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
                np.float32)

        square_dx = (self.robot_state[0] - self.object_state[0])**2
        square_dy = (self.robot_state[1] - self.object_state[1])**2
        square_dz = (self.robot_state[2] - self.object_state[2])**2

        self.distance = sqrt(square_dx + square_dy + square_dz)

        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]

        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)

        if terminated:
            reward = -0.1
            self.terminated = True

        elif self.step_counter > self.max_steps_one_episode:
            reward = -0.1
            self.terminated = True

        elif self.distance < 0.1:
            reward = 1
            self.terminated = True
        else:
            reward = 0
            self.terminated = False

        info = {'distance:', self.distance}
        (_, _, px, _,
         _) = p.getCameraImage(width=960,
                               height=960,
                               viewMatrix=self.view_matrix,
                               projectionMatrix=self.projection_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL)
        self.images = px
        self.processed_image = self._process_image(self.images)
        self.observation = self.object_state
        return self.processed_image, reward, self.terminated, info

    def close(self):
        p.disconnect()

    def run_for_debug(self, target_position):
        temp_robot_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=target_position,
            targetOrientation=self.orientation,
            jointDamping=self.joint_damping,
        )
        for i in range(1, self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=temp_robot_joint_positions[i - 1],
            )
        p.stepSimulation()

        if self.is_good_view:
            time.sleep(0.05)

        return self._get_force_sensor_value()

    def _get_force_sensor_value(self):
        force_sensor_value = p.getJointState(bodyUniqueId=self.kuka_id,
                                             jointIndex=self.num_joints -
                                             1)[2][2]
        return force_sensor_value


class CustomSkipFrame(gym.Wrapper):
    """ Make a 4 frame skip, so the observation space will change to (4,84,84) from (1,84,84)
    Args:
        gym ([type]): [description]
    """
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = spaces.Box(low=0,
                                            high=1,
                                            shape=(skip, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        states = []
        state, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                states.append(state)
            else:
                states.append(state)
        states = np.concatenate(states, 0)[None, :, :, :]
        return states.astype(np.float32), reward, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)],
                                0)[None, :, :, :]
        return states.astype(np.float32)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    env = KukaCamReachEnv(is_good_view=False, is_render=True)
    env = CustomSkipFrame(env)

    obs = env.reset()
    print(obs)
    print(obs.shape)

    sum_reward = 0
    for i in range(10):
        obs=env.reset()
        img = obs[0][0]
        plt.imshow(img, cmap='gray')
        plt.show()
        for i in range(2000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            #  print("i={},\naction={},\nobs={},\ndone={},\n".format(i,action,obs,done,))
            print(colored("reward={},info={}".format(reward, info), "cyan"))
            sum_reward += reward
            if done:
                break
        # time.sleep(0.1)
    print()
    print(sum_reward)
