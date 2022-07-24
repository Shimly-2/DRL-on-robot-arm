#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        main.py
 @Created:          2022/7/18 10:56
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
import torch
import numpy as np
from tqdm import tqdm
import random
import envs
from envs.rl_reach_env import RLReachEnv
from envs.rl_camreach_env import RLCamReachEnv, CustomSkipFrame
from envs.rl_push_env import RLPushEnv
from envs.rl_pick_env import RLPickEnv
from utils.rl_utils import Trajectory, ReplayBuffer_Trajectory_push, ReplayBuffer_Trajectory_reach, ReplayBuffer_Trajectory_reach_CNN
from config import opt
from utils.visualize import Visualizer
import algo
from algo.TD3.TD3_mlp import TD3_MLP
from algo.TD3.TD3_cnn import TD3_CNN
from algo.DDPG.DDPG_mlp import DDPG_MLP
from algo.DARC.DARC_mlp import DARC_MLP
import json
import pandas as pd


def eval_policy(policy, env_name, seed, eval_episodes=25, eval_cnt=None):
    eval_env = RLReachEnv(is_render=True, is_good_view=False)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    rate = 0
    with tqdm(total=int(eval_episodes), desc='[Evaluation %d]' % eval_cnt) as pbar:
        for episode_idx in range(eval_episodes):
            state, done = eval_env.reset(), False
            while not done:
                action = policy.take_action(np.array(state))
                next_state, reward, done, is_success = eval_env.step(action)

                avg_reward += reward
                state = next_state
                if is_success == True:
                    rate = rate + 1
            pbar.update(1)
        avg_reward /= eval_episodes
        rate /= eval_episodes
        pbar.set_postfix({
            'episode': '%d' % eval_episodes,
            'avg return': '%.3f' % avg_reward,
            'success rata': '%.2f' % rate
        })
    # print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))
    eval_env.close()
    return avg_reward, rate


def run(**kwargs):
    # 根据命令行参数更新配置
    opt._parse(kwargs)
    # 连接visdom
    vis = Visualizer(opt.vis_name, port=opt.vis_port)
    # 建立env
    env = getattr(envs, opt.env)(is_render=True, is_good_view=False)
    # 获取环境中的动作空间、状态空间、动作限幅
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0]) + 0.3
    # 随机种子使结果可复现
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    # 生成经验回放数组
    replay_buffer = ReplayBuffer_Trajectory_reach(opt.buffer_size)
    # 建立算法agent
    agent = getattr(algo, opt.algo)(state_dim=state_dim, action_dim=action_dim, action_bound=action_bound)

    return_list = []
    max_rate = 0
    # 我们定义的num_episodes为500，这里分为100个iteration，每个iteration有50个episode，然后通过进度条显示训练过程
    for i in range(100):
        with tqdm(total=int(opt.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            rate = 0.0
            for i_episode in range(int(opt.num_episodes / 10)):
                episode_return = 0
                a = []
                initial_a = [0.5320540070533752, -0.0011213874677196145, 0.4962984025478363]
                a.append(initial_a)
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    # if i == 0:
                    #     action = env.action_space.sample()
                    # else:
                    action = agent.take_action(state)
                    action = (action + np.random.normal(0, action_bound * opt.gamma, size=action_dim)).clip(
                        -action_bound, action_bound)
                    new_a = [initial_a[0] + action[0] * opt.reach_ctr,
                             initial_a[1] + action[1] * opt.reach_ctr,
                             initial_a[2] + action[2] * opt.reach_ctr]
                    a.append(new_a)
                    initial_a = new_a
                    # action = (action + np.random.normal(0, 1 * opt.gamma, size=action_dim))
                    state, reward, done, is_success = env.step(action)
                    if is_success == True:
                        rate = rate + 1
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                vis.plot("return", episode_return)
                a.append(state[-3:])
                vis.scatter_reach('action', a)
                # vis.scatter_updata(win, state[-3:])
                return_list.append(episode_return)
                if replay_buffer.size() >= opt.minimal_episodes:
                    for _ in range(opt.n_train):
                        transition_dict = replay_buffer.sample(opt.batch_size, use_her=True, her_ratio=opt.her_ratio)
                        agent.train(transition_dict)
                        # vis.plot("loss", loss)
                # if (i_episode + 1) % 10 == 0:
                #     pbar.set_postfix({
                #         'episode':
                #             '%d' % (opt.num_episodes / 10 * i + i_episode + 1),
                #         'return':
                #             '%.3f' % np.mean(return_list[-10:])
                #     })
                #     vis.plot("avg_return", np.mean(return_list[-10:]))
                if (i_episode + 1) % 25 == 0:
                    rate = rate / 25.0
                    vis.plot("success_rate", rate)
                    if rate >= max_rate:
                        agent.save(f"D:/DRL_Diana_robot_arm/result/{rate}")
                        max_rate = rate
                        opt.her_ratio = opt.her_ratio * 0.75
                    pbar.set_postfix({
                        'episode': '%d' % (opt.num_episodes / 10 * i + i_episode + 1),
                        'avg return': '%.3f' % np.mean(return_list[-25:]),
                        'success rata': '%.2f' % rate
                    })
                    vis.plot("avg_return", np.mean(return_list[-25:]))
                    rate = 0.0
                pbar.update(1)


def train_reach_with_TD3(**kwargs):
    # 根据命令行参数更新配置
    opt._parse(kwargs)

    vis = Visualizer(opt.vis_name, port=opt.vis_port)
    env = RLReachEnv(is_render=True, is_good_view=False)
    state_dim = 6
    action_dim = 3
    action_bound = float(env.action_space.high[0]) + 0.3

    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    # env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory_reach(opt.buffer_size)
    agent = TD3_MLP(state_dim, action_dim, action_bound, opt.hidden_dim, opt.actor_lr,
                 opt.critic_lr, opt.sigma, opt.tau, opt.gamma, opt.policy_noise, opt.noise_clip, opt.policy_freq, opt.device)

    return_list = []
    max_rate = 0
    for i in range(100):
        with tqdm(total=int(opt.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            rate = 0.0
            for i_episode in range(int(opt.num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    # if i == 0:
                    #     action = env.action_space.sample()
                    # else:
                    action = agent.take_action(state)
                    # action = (action + np.random.normal(0, action_bound * expl_noise, size=action_dim)).clip(
                    #     -action_bound, action_bound)
                    action = (action + np.random.normal(0, 1 * opt.gamma, size=action_dim))
                    state, reward, done, is_success = env.step(action)
                    if is_success == True:
                        rate = rate + 1
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                vis.plot("return", episode_return)
                return_list.append(episode_return)
                if replay_buffer.size() >= opt.minimal_episodes:
                    for _ in range(opt.n_train):
                        transition_dict = replay_buffer.sample(opt.batch_size, use_her=True, her_ratio=opt.her_ratio)
                        agent.train(transition_dict)
                        # vis.plot("loss", loss)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (opt.num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                    vis.plot("avg_return", np.mean(return_list[-10:]))
                if (i_episode + 1) % 25 == 0:
                    rate = rate / 25.0
                    vis.plot("success_rate", rate)
                    if rate >= max_rate:
                        agent.save(f"D:/DRL_Diana_robot_arm/result/{rate}")
                        max_rate = rate
                        opt.her_ratio = opt.her_ratio * 0.75
                        # opt.expl_noise = opt.expl_noise * 0.75
                    rate = 0.0
                pbar.update(1)


def train_reach_with_DARC(**kwargs):
    # 根据命令行参数更新配置
    opt._parse(kwargs)

    vis = Visualizer(opt.vis_name, port=opt.vis_port)
    env = RLReachEnv(is_render=True, is_good_view=False)
    state_dim = 6
    action_dim = 3
    action_bound = float(env.action_space.high[0]) + 0.3

    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    # env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory_reach(opt.buffer_size)
    agent = DARC_MLP(state_dim, action_dim, action_bound, opt.hidden_dim, opt.actor_lr,
                 opt.critic_lr, opt.sigma, opt.tau, opt.gamma, opt.policy_noise, opt.noise_clip,
                 opt.policy_freq, opt.q_weight, opt.regularization_weight, opt.device)

    return_list = []
    max_rate = 0
    for i in range(100):
        with tqdm(total=int(opt.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            rate = 0.0
            for i_episode in range(int(opt.num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    # if i == 0:
                    #     action = env.action_space.sample()
                    # else:
                    action = agent.take_action(state)
                    # action = (action + np.random.normal(0, action_bound * expl_noise, size=action_dim)).clip(
                    #     -action_bound, action_bound)
                    action = (action + np.random.normal(0, 1 * opt.gamma, size=action_dim))
                    state, reward, done, is_success = env.step(action)
                    if is_success == True:
                        rate = rate + 1
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                vis.plot("return", episode_return)
                return_list.append(episode_return)
                if replay_buffer.size() >= opt.minimal_episodes:
                    for _ in range(opt.n_train):
                        transition_dict = replay_buffer.sample(opt.batch_size, use_her=True, her_ratio=opt.her_ratio)
                        agent.train(transition_dict)
                        # vis.plot("loss", loss)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (opt.num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                    vis.plot("avg_return", np.mean(return_list[-10:]))
                if (i_episode + 1) % 25 == 0:
                    rate = rate / 25.0
                    vis.plot("success_rate", rate)
                    if rate >= max_rate:
                        agent.save(f"D:/DRL_Diana_robot_arm/result/{rate}")
                        max_rate = rate
                        opt.her_ratio = opt.her_ratio * 0.75
                        # opt.expl_noise = opt.expl_noise * 0.75
                    pbar.set_postfix({
                        'episode': '%d' % (opt.num_episodes / 10 * i + i_episode + 1),
                        'avg return': '%.3f' % np.mean(return_list[-25:]),
                        'success rata': '%.2f' % rate
                    })
                    vis.plot("avg_return", np.mean(return_list[-10:]))
                    rate = 0.0
                pbar.update(1)


def train_reach_with_DDPG(**kwargs):
    # 根据命令行参数更新配置
    opt._parse(kwargs)

    vis = Visualizer(opt.vis_name, port=opt.vis_port)
    env = RLReachEnv(is_render=True, is_good_view=False)
    state_dim = 6
    action_dim = 3
    action_bound = float(env.action_space.high[0]) + 0.3

    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    # env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory_reach(opt.buffer_size)
    agent = DDPG_MLP(state_dim, action_dim, action_bound, opt.hidden_dim, opt.actor_lr,
                 opt.critic_lr, opt.sigma, opt.tau, opt.gamma, opt.device)

    return_list = []
    max_rate = 0
    for i in range(100):
        with tqdm(total=int(opt.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            rate = 0.0
            for i_episode in range(int(opt.num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    # if i == 0:
                    #     action = env.action_space.sample()
                    # else:
                    action = agent.take_action(state)
                    # action = (action + np.random.normal(0, action_bound * expl_noise, size=action_dim)).clip(
                    #     -action_bound, action_bound)
                    action = (action + np.random.normal(0, 1 * opt.gamma, size=action_dim))
                    state, reward, done, is_success = env.step(action)
                    if reward == 0:
                        rate = rate + 1
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                vis.plot("return", episode_return)
                return_list.append(episode_return)
                if replay_buffer.size() >= opt.minimal_episodes:
                    for _ in range(opt.n_train):
                        transition_dict = replay_buffer.sample(opt.batch_size, use_her=True, her_ratio=opt.her_ratio)
                        agent.train(transition_dict)
                        # vis.plot("loss", loss)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (opt.num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                    vis.plot("avg_return", np.mean(return_list[-10:]))
                if (i_episode + 1) % 25 == 0:
                    rate = rate / 25.0
                    vis.plot("success_rate", rate)
                    if rate >= max_rate:
                        agent.save(f"D:/DRL_Diana_robot_arm/result/{rate}")
                        max_rate = rate
                        opt.her_ratio = opt.her_ratio * 0.75
                        # opt.expl_noise = opt.expl_noise * 0.75
                    rate = 0.0
                pbar.update(1)


def train_reach_with_TD3_CNN(**kwargs):
    # 根据命令行参数更新配置
    opt._parse(kwargs)

    vis = Visualizer(opt.vis_name, port=opt.vis_port)
    env = RLCamReachEnv(is_render=True, is_good_view=False)
    env = CustomSkipFrame(env)
    state_dim = (4, 84, 84)
    action_dim = 3
    action_bound = float(env.action_space.high[0]) + 0.3

    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    # env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory_reach_CNN(opt.buffer_size)
    agent = TD3_CNN(state_dim, action_dim, action_bound, opt.hidden_dim, opt.actor_lr,
                 opt.critic_lr, opt.sigma, opt.tau, opt.gamma, opt.policy_noise, opt.noise_clip, opt.policy_freq, opt.device)

    return_list = []
    max_rate = 0
    for i in range(100):
        with tqdm(total=int(opt.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            rate = 0.0
            for i_episode in range(int(opt.num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    # if i == 0:
                    #     action = env.action_space.sample()
                    # else:
                    action = agent.take_action(state)
                    # action = (action + np.random.normal(0, action_bound * expl_noise, size=action_dim)).clip(
                    #     -action_bound, action_bound)
                    action = (action + np.random.normal(0, 1 * opt.gamma, size=action_dim))
                    state, reward, done, is_success = env.step(action)
                    if reward == 0:
                        rate = rate + 1
                    episode_return += reward
                    state = list(state)
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                vis.plot("return", episode_return)
                return_list.append(episode_return)
                if replay_buffer.size() >= opt.minimal_episodes:
                    for _ in range(opt.n_train):
                        transition_dict = replay_buffer.sample(opt.batch_size, use_her=False, her_ratio=opt.her_ratio)
                        agent.train(transition_dict)
                        # vis.plot("loss", loss)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (opt.num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                    vis.plot("avg_return", np.mean(return_list[-10:]))
                if (i_episode + 1) % 25 == 0:
                    rate = rate / 25.0
                    vis.plot("success_rate", rate)
                    if rate >= max_rate:
                        agent.save(f"D:/DRL_Diana_robot_arm/result/{rate}")
                        max_rate = rate
                        opt.her_ratio = opt.her_ratio * 0.75
                        # opt.expl_noise = opt.expl_noise * 0.75
                    rate = 0.0
                pbar.update(1)


def train_push_with_TD3(**kwargs):
    # 根据命令行参数更新配置
    opt._parse(kwargs)

    vis = Visualizer(opt.vis_name, port=opt.vis_port)
    env = RLPushEnv(is_render=True, is_good_view=False)
    state_dim = 9
    action_dim = 3
    action_bound = float(env.action_space.high[0])

    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    # env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory_push(opt.buffer_size)
    agent = TD3_MLP(state_dim, action_dim, action_bound, opt.hidden_dim, opt.actor_lr,
                 opt.critic_lr, opt.sigma, opt.tau, opt.gamma, opt.policy_noise, opt.noise_clip, opt.policy_freq, opt.device)

    return_list = []
    max_rate = 0
    for i in range(100):
        with tqdm(total=int(opt.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            rate = 0.0
            for i_episode in range(int(opt.num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    # if i == 0:
                    #     action = env.action_space.sample()
                    # else:
                    action = agent.take_action(state)
                    # action = (action + np.random.normal(0, action_bound * expl_noise, size=action_dim)).clip(
                    #     -action_bound, action_bound)
                    action = (action + np.random.normal(0, action_bound * opt.gamma, size=action_dim))
                    state, reward, done, info = env.step(action)
                    if reward == 100:
                        rate = rate + 1
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                vis.plot("return", episode_return)
                return_list.append(episode_return)
                if replay_buffer.size() >= opt.minimal_episodes:
                    for _ in range(opt.n_train):
                        transition_dict = replay_buffer.sample(opt.batch_size, use_her=True, her_ratio=opt.her_ratio)
                        agent.train(transition_dict)
                        # vis.plot("loss", loss)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (opt.num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                    vis.plot("avg_return", np.mean(return_list[-10:]))
                if (i_episode + 1) % 25 == 0:
                    rate = rate / 25.0
                    vis.plot("success_rate", rate)
                    if rate >= max_rate:
                        agent.save(f"D:/DRL_Diana_robot_arm/result/push/{rate}")
                        max_rate = rate
                        opt.her_ratio = opt.her_ratio * 0.75
                        # opt.expl_noise = opt.expl_noise * 0.75
                    rate = 0.0
                pbar.update(1)


def train_pick_with_TD3(**kwargs):
    # 根据命令行参数更新配置
    opt._parse(kwargs)

    vis = Visualizer(opt.vis_name, port=opt.vis_port)
    env = RLPickEnv(is_render=True, is_good_view=False)
    state_dim = 9
    action_dim = 3
    action_bound = float(env.action_space.high[0])

    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    # env = WorldEnv()
    replay_buffer = ReplayBuffer_Trajectory_push(opt.buffer_size)
    agent = TD3_MLP(state_dim, action_dim, action_bound, opt.hidden_dim, opt.actor_lr,
                 opt.critic_lr, opt.sigma, opt.tau, opt.gamma, opt.policy_noise, opt.noise_clip, opt.policy_freq, opt.device)

    return_list = []
    max_rate = 0
    for i in range(100):
        with tqdm(total=int(opt.num_episodes / 10), desc='Iteration %d' % i) as pbar:
            rate = 0.0
            for i_episode in range(int(opt.num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                traj = Trajectory(state)
                done = False
                while not done:
                    # if i == 0:
                    #     action = env.action_space.sample()
                    # else:
                    action = agent.take_action(state)
                    # action = (action + np.random.normal(0, action_bound * expl_noise, size=action_dim)).clip(
                    #     -action_bound, action_bound)
                    action = (action + np.random.normal(0, action_bound * opt.gamma, size=action_dim))
                    state, reward, done, info = env.step(action)
                    if reward == 100:
                        rate = rate + 1
                    episode_return += reward
                    traj.store_step(action, state, reward, done)
                replay_buffer.add_trajectory(traj)
                vis.plot("return", episode_return)
                return_list.append(episode_return)
                if replay_buffer.size() >= opt.minimal_episodes:
                    for _ in range(opt.n_train):
                        transition_dict = replay_buffer.sample(opt.batch_size, use_her=True, her_ratio=opt.her_ratio)
                        agent.train(transition_dict)
                        # vis.plot("loss", loss)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (opt.num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                    vis.plot("avg_return", np.mean(return_list[-10:]))
                if (i_episode + 1) % 25 == 0:
                    rate = rate / 25.0
                    vis.plot("success_rate", rate)
                    if rate >= max_rate:
                        agent.save(f"D:/DRL_Diana_robot_arm/result/push/{rate}")
                        max_rate = rate
                        opt.her_ratio = opt.her_ratio * 0.75
                        # opt.expl_noise = opt.expl_noise * 0.75
                    rate = 0.0
                pbar.update(1)


def get_vis_data():
    """解析visdom得到的json文件"""
    key_list = []
    def getJsonKey(json_data):
        # 递归获取字典中所有key
        for key in json_data.keys():
            if type(json_data[key]) == type({}):
                getJsonKey(json_data[key])
            key_list.append(key)
        return key_list

    print('-------------------------------------------------------------------')
    print('==> Start processing json data..')

    with open(opt.jsonfile,'r') as jsonfile:
        # 将json文件导入为dict
        jsondata = json.load(jsonfile)
        # 获取key列表
        key_list = getJsonKey(jsondata)
        # print(key_list)
        # print(jsondata.keys())
        # print(jsondata['jsons'].keys())
        # 获取所有图表的keys
        keys = list(jsondata['jsons'].keys())
        # print(keys)
        # 图表的x，y轴数据存放在以下dict中
        # print(jsondata['jsons'][keys[0]]['content']['data'][0].keys())
        # 下面开始获取所有的图表数据并存入csv中
        for key in keys:
            xData = jsondata['jsons'][key]['content']['data'][0]['x']
            yData = jsondata['jsons'][key]['content']['data'][0]['y']
            dict = {'xData': xData, 'yData': yData}
            targetname = opt.csvname + key + ".csv"
            df = pd.DataFrame(dict)
            df.to_csv(targetname, index=False)
            print('{:<60}{:<15}'.format(str(targetname), 'write successfully'))  # {:<30d}含义是 左对齐，且占用30个字符位
    print('-------------------------------------------------------------------')


def help(**kwargs):
    """
    打印帮助的信息： python file.py help
    """
    print("""-------------------------------------------------------------------
    usage : python file.py <function> [--args=value]
    <function> := train_reach_with_TD3 | train_reach_with_DDPG | train_reach_with_TD3_CNN | train_push_with_TD3 | train_pick_with_TD3 | get_vis_data | help 
            [train_reach_with_TD3]        --- Start train model
            [train_reach_with_DDPG]       --- Start test for random imgages
            [train_reach_with_TD3_CNN]    --- Start test for total acc
            [train_push_with_TD3]         --- Start test for write results to csv
            [train_pick_with_TD3]         --- Start analysis for datasets
            [get_vis_data]                --- Start make divide for datasets
    example: 
            python {0} train --env='TD3' --lr=0.01
            python {0} testforacc --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))
    opt._parsehelp(kwargs)

    # from inspect import getsource
    # source = (getsource(opt.__class__))
    # print(source)
    print('-------------------------------------------------------------------')


if __name__=='__main__':
    import fire
    fire.Fire()
    # train_reach_with_TD3_CNN()
