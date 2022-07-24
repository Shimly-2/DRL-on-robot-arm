#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        DQN_mlp.py
 @Created:          2022/7/18 10:52
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
import torch.nn.functional as F
import copy
from algo.DQN.net_mlp import QNet
from config import opt
import numpy as np

class DQN_MLP:
    '''DDPG算法'''
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim=opt.hidden_dim,
                 actor_lr=opt.actor_lr, gamma=opt.gamma, epsilon=opt.epsilon,
                 target_update=opt.target_update, device=opt.device):
        '''
        用于初始化DDPG算法中的各项参数，
        初始化策略网络与估值网络

        Args:
            state_dim (int):       状态空间维数
            hidden_dim (int):      隐藏层大小
            action_dim (int):      动作空间维数
            action_bound (float):  动作空间限幅
            actor_lr (float):      策略网络学习率
            critic_lr (float):     估值网络学习率
            sigma (float):         高斯噪声的标准差
            tau (float):           目标网络软更新参数
            gamma (float):         折扣因子
            device (any):          训练设备

        Returns:
            None
        '''
        self.action_dim = action_dim

        self.q_net = QNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_q_net = QNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.q_net_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=actor_lr)

        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.action_bound = action_bound
        self.device = device
        self.total_it = 0

    def take_action(self, state):
        '''
        由策略网络选择动作，
        并加入高斯噪声增加探索效率

        Args:
            state (array):  当前智能体状态

        Returns:
            action (array): 智能体的下一步动作
        '''
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def train(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + (1 - dones) * self.gamma * max_next_q_values # TD误差目标

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数

        self.q_net_optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.q_net_optimizer.step()

        if self.total_it % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.total_it += 1

    def save(self, filename):
        torch.save(self.q_net.state_dict(), filename + "_q_net.pt")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pt")

    def load(self, filename):
        self.q_net.load_state_dict(torch.load(filename + "_q_net.pt"))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pt"))
        self.target_q_net = copy.deepcopy(self.q_net)