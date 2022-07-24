#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        rl_push_env.py
 @Created:          2022/7/18 14:59
 @Software:         PyCharm
 
 @Author:           HHH
 @Email:            1950049@tongji.edu.cn
 @Gitee:            https://gitee.com/jin-yiyang
 @Version:          v1.0
 
 @Description:      Main Function:    
                    
 @Function List:    exit() -- exit the interpreter by raising SystemExit
                    getdlopenflags() -- returns flags to be used for dlopen() calls
                    getprofile() -- get the global profiling function
 ------------------------------------------------------------------
 @Change History :                                                          
  <Date>     | <Version> | <Author>       | <Description>                   
 ------------------------------------------------------------------
  2022/7/18   | v1.0      | HHH            | Create file                     
 ------------------------------------------------------------------
'''
import numpy as np
import pybullet as p
import pybullet_data
import os
import gym
from gym import spaces
from gym.utils import seeding
import random
import time
import math
from config import opt

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    # np.linalg.norm指求范数，默认是l2范数
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class RLPushEnv(gym.Env):
    """创建强化学习机械臂push任务仿真环境"""
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    def __init__(self, is_render=False, is_good_view=False):
        """
        用于初始化push环境中的各项参数，

        Args:
            is_render (bool):       是否创建场景可视化
            is_good_view (bool):    是否创建更优视角

        Returns:
            None
        """
        self.is_render = is_render
        self.is_good_view = is_good_view
        self.max_steps_one_episode = opt.max_steps_one_episode

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # 机械臂移动范围限制
        self.x_low_obs = 0.2
        self.x_high_obs = 0.7
        self.y_low_obs = -0.3
        self.y_high_obs = 0.3
        self.z_low_obs = 0
        self.z_high_obs = 0.55

        # 机械臂动作范围限制
        self.x_low_action = -0.4
        self.x_high_action = 0.4
        self.y_low_action = -0.4
        self.y_high_action = 0.4
        self.z_low_action = -0.6
        self.z_high_action = 0.3

        # 被推动物体与目标距离的距离阈值
        self.distance_threshold = 0.05

        # 设置相机
        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([self.x_low_action, self.y_low_action, self.z_low_action]),
            high=np.array([self.x_high_action, self.y_high_action, self.z_high_action]),
            dtype=np.float32)

        # 状态空间
        self.observation_space = spaces.Box(
            low=np.array([self.x_low_obs, self.y_low_obs, self.z_low_obs]),
            high=np.array([self.x_high_obs, self.y_high_obs, self.z_high_obs]),
            dtype=np.float32)

        # 时间步计数器
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

        # 初始关节角度
        self.init_joint_positions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684,
            -0.006539
        ]

        self.orientation = p.getQuaternionFromEuler(
            [0., -math.pi, math.pi / 2.])

        self.last_object_pos = 0.0
        self.last_target_pos = 0.0
        self.current_object_pos = 0.0
        self.current_target_pos = 0.0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        """随机种子"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """环境reset，获得初始state"""
        # 初始化时间步计数器
        self.step_counter = 0

        p.resetSimulation()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # 初始化重力以及运行结束标志
        self.terminated = False
        p.setGravity(0, 0, -10)

        # 状态空间的限制空间可视化，以白线标识
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

        # 载入平面
        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),basePosition=[0, 0, -0.65])
        # 载入机械臂
        self.kuka_id = p.loadURDF(os.path.join(self.urdf_root_path, "kuka_iiwa/model.urdf"), useFixedBase=True)
        # 载入桌子
        p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"), basePosition=[0.5, 0, -0.65])
        # p.loadURDF(os.path.join(self.urdf_root_path, "tray/traybox.urdf"),basePosition=[0.55,0,0])
        # object_id=p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=[0.53,0,0.02])

        # 通过循环筛选出满足距离条件的随机点
        xpos, ypos, zpos, ang, orn, xpos_target, ypos_target, zpos_target, ang_target, orn_target = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for _ in range(1000):
            # 物体的位姿
            xpos = random.uniform(self.x_low_obs, self.x_high_obs)
            ypos = random.uniform(self.y_low_obs, self.y_high_obs)
            zpos = 0.01   # TODO 原z=0.01
            ang = 3.14 * 0.5 + 3.1415925438 * random.random()
            orn = p.getQuaternionFromEuler([0, 0, ang])

            # 目标物体的位姿
            xpos_target = random.uniform(self.x_low_obs, self.x_high_obs)
            ypos_target = random.uniform(self.y_low_obs, self.y_high_obs)
            zpos_target = 0.01  # TODO 原z=0.01
            ang_target = 3.14 * 0.5 + 3.1415925438 * random.random()
            orn_target = p.getQuaternionFromEuler([0, 0, ang_target])

            # 确保距离在一定范围内
            self.dis_between_target_block = math.sqrt(
                (xpos - xpos_target) ** 2 + (ypos - ypos_target) ** 2 + (zpos - zpos_target) ** 2)
            if self.dis_between_target_block >= 0.22 and self.dis_between_target_block <= 0.25:
                break

        # 载入物体
        self.object_id = p.loadURDF("../models/cube_small_push.urdf",
                                    basePosition=[xpos, ypos, zpos],
                                    baseOrientation=[orn[0], orn[1], orn[2], orn[3]])
        # 载入目标物体
        self.target_object_id = p.loadURDF("../models/cube_small_target_push.urdf",
                                    basePosition=[xpos_target, ypos_target, zpos_target],
                                    baseOrientation=[orn_target[0], orn_target[1], orn_target[2], orn_target[3]],
                                    useFixedBase=1)
        # 避免碰撞检测
        p.setCollisionFilterPair(self.target_object_id, self.object_id, -1, -1, 0)

        # 关节角初始化
        self.num_joints = p.getNumJoints(self.kuka_id)
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )
        # for i in range(self.num_joints):
        #      print(p.getJointInfo(self.kuka_id, i))

        self.robot_pos_obs = p.getLinkState(self.kuka_id,
                                            self.num_joints - 1)[4]
        # logging.debug("init_pos={}\n".format(p.getLinkState(self.kuka_id,self.num_joints-1)))
        p.stepSimulation()
        obs = self._get_obs()
        self.last_object_pos = obs[3:6]
        self.last_target_pos = obs[-3:]
        # self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        # self.object_pos = self.robot_pos_obs
        #
        # goal = [random.uniform(self.x_low_obs, self.x_high_obs),
        #         random.uniform(self.y_low_obs, self.y_high_obs),
        #         random.uniform(self.z_low_obs, self.z_high_obs)]
        # self.object_state = np.array(
        #     p.getBasePositionAndOrientation(self.object_id)[0]).astype(
        #     np.float32)
        # return np.array(self.object_pos).astype(np.float32), self.object_state
        return obs

    def _get_obs(self):
        """获取当前的观测：机械臂末端坐标、物体坐标、目标位置坐标"""
        # 关于机械臂的状态观察，可以从以下几个维度进行考虑
        # 末端位置、夹持器状态位置、物体位置、物体姿态、  物体相对末端位置、物体线速度、物体角速度、末端速度、物体相对末端线速度
        # 末端位置 3vec 及速度
        robot_obs = p.getLinkState(self.kuka_id, self.num_joints - 1, computeLinkVelocity=1)
        robotPos = np.array(robot_obs[4])
        robotOrn_temp = np.array(robot_obs[5])
        robot_linear_Velocity = np.array(robot_obs[6])
        robot_angular_Velocity = np.array(robot_obs[7])
        # 把四元数转换成欧拉角，使数据都是三维的
        robotOrn = p.getEulerFromQuaternion(robotOrn_temp)
        robotOrn = np.array(robotOrn)
        # 物体位置、姿态
        blockPos, blockOrn_temp = p.getBasePositionAndOrientation(self.object_id)
        blockPos = np.array(blockPos)
        blockOrn = p.getEulerFromQuaternion(blockOrn_temp)
        blockOrn = np.array(blockOrn)
        # 物体相对位置 vec *3
        relative_pos = blockPos - robotPos
        # relative_orn = blockOrn - gripperOrn
        # 物体的线速度和角速度
        block_Velocity = p.getBaseVelocity(self.object_id)
        block_linear_velocity = np.array(block_Velocity[0])
        target_pos = np.array(p.getBasePositionAndOrientation(self.target_object_id)[0])
        block_angular_velocity = np.array(block_Velocity[1])
        # 物体相对末端线速度
        # block_relative_linear_velocity = block_linear_velocity - gripper_linear_Velocity

        # 问题：是否把相对速度、相对位置想得过于理所当然了？ 用不用进行四元数的转换、需不需要考虑位姿下的相对位置，直接写可行吗？
        # blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
        # #we return the relative x,y position and euler angle of block in gripper space
        # blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]
        # self._observation.extend(list(blockInGripperPosXYEulZ))
        # to numpy.array()
        #
        # obs = [
        #     robotPos.flatten(),
        #     relative_pos.flatten()
        # ]
        # print(blockPos)

        achieved_goal = blockPos.copy()
        # end_pos = []
        # for i in range(1, len(obs)):
        #     end_pos = np.append(end_pos, obs[i])
        # obs = end_pos.reshape(-1)
        # print(obs,achieved_goal, target_pos)

        # self._observation = obs
        return np.hstack((np.array(robotPos).astype(np.float32), achieved_goal, target_pos))

    def step(self, action):
        """根据action获取下一步环境的state、reward、done"""
        limit_x = [0.2, 0.7]
        limit_y = [-0.3, 0.3]
        limit_z = [0, 0.1]

        def clip_val(val, limit):
            if val < limit[0]:
                return limit[0]
            if val > limit[1]:
                return limit[1]
            return val
        dv = 0.08
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv

        # 获取当前机械臂末端坐标
        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        # 计算下一步的机械臂末端坐标
        self.new_robot_pos = [
            clip_val(self.current_pos[0] + dx,limit_x), clip_val(self.current_pos[1] + dy,limit_y),
            clip_val(self.current_pos[2] + dz,limit_z)
        ]
        # 通过逆运动学计算机械臂移动到新位置的关节角度
        self.robot_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=[self.new_robot_pos[0], self.new_robot_pos[1], self.new_robot_pos[2]],
            targetOrientation=self.orientation,
            jointDamping=self.joint_damping,
        )
        # 使机械臂移动到新位置
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )
        p.stepSimulation()

        # 在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1
        return self._reward()

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        # print(achieved_goal, goal, d)

        # if self.reward_type == 'sparse':
        return -(d > self.distance_threshold).astype(np.float32)
        # else:
        # return -d*10

    def _reward(self):
        """根据state计算当前的reward"""
        # 获取机械臂当前的末端坐标
        # 一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        # self.object_state=p.getBasePositionAndOrientation(self.object_id)
        # self.object_state=np.array(self.object_state).astype(np.float32)

        # 获取物体当前的位置坐标
        self.object_state = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
                np.float32)
        # 获取目标物体当前的位置坐标
        self.target_state = np.array(
            p.getBasePositionAndOrientation(self.target_object_id)[0]).astype(
            np.float32)
        # 获取当前的观测
        obs = self._get_obs()

        # 计算物体与目标物体间的距离是否发生改变
        self.current_object_pos = obs[3:6]
        self.current_target_pos = obs[-3:]
        self.distance_current = np.linalg.norm(self.current_object_pos - self.current_target_pos, axis=-1)
        self.distance_last = np.linalg.norm(self.last_object_pos - self.last_target_pos, axis=-1)
        test = self.distance_current - self.distance_last
        if abs(test) < 1e-5:
            test = 0.01
        # print("test:",test)
        self.last_object_pos = self.current_object_pos
        self.last_target_pos = self.current_target_pos

        # 用机械臂末端和物体的距离作为奖励函数的依据
        self.distance_target = np.linalg.norm(self.object_state - self.target_state, axis=-1)
        self.distance_robot = np.linalg.norm(self.robot_state - self.object_state, axis=-1)
        # print(self.distance)

        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]

        # 如果机械比末端超过了obs的空间，也视为done，而且会给予一定的惩罚
        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)

        # if terminated:
        #     reward = -50.0
        #     self.terminated = True

        # 如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        if self.step_counter > self.max_steps_one_episode:
            reward = -self.distance_target * 50
            self.terminated = True

        elif self.distance_target < 0.05:
            reward = 100 # 10.0
            self.terminated = True
        else:
            # reward = -1 # -0.1
            reward = - test * 100
            self.terminated = False

        info = {
            'is_success': self._is_success(obs[3:6], obs[-3:]),
        }
        # self.observation=self.robot_state
        self.observation = self.object_state
        self.observation = self.robot_state

        goal = [random.uniform(self.x_low_obs,self.x_high_obs),
                random.uniform(self.y_low_obs,self.y_high_obs),
                random.uniform(self.z_low_obs, self.z_high_obs)]
        return obs, reward, self.terminated, info

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        # self.distance_threshold = 0.05
        return (d < self.distance_threshold).astype(np.float32)

    def close(self):
        p.disconnect()

if __name__ == '__main__':
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数
    env = RLPushEnv(is_good_view=True, is_render=True)
    print('env={}'.format(env))
    print(env.observation_space.shape)
    # print(env.observation_space.sample())
    # print(env.action_space.sample())
    print(env.action_space.shape)
    obs = env.reset()
    # print(Fore.RED + 'obs={}'.format(obs))
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print('obs={},reward={},done={}'.format(obs, reward, done))

    sum_reward = 0
    success_times = 0
    for i in range(100):
        env.reset()
        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print('reward={},done={}'.format(reward, done))
            sum_reward += reward
            if reward == 1:
                success_times += 1
            if done:
                break
        # time.sleep(0.1)
    print()
    print('sum_reward={}'.format(sum_reward))
    print('success rate={}'.format(success_times / 50))
