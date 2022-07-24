#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        DATD3_mlp.py
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
from algo.DATD3.net_mlp import PolicyNet, QValueNet
from config import opt
import numpy as np

class DATD3_MLP:
    ''' DATD3算法 '''
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim=opt.hidden_dim,
                 actor_lr=opt.actor_lr, critic_lr=opt.critic_lr, sigma=opt.sigma, tau=opt.tau, gamma=opt.gamma,
                 policy_noise=opt.policy_noise, noise_clip=opt.noise_clip, policy_freq=opt.policy_freq, device=opt.device):
        '''
        用于初始化TD3算法中的各项参数，
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
            policy_noise (float):  策略噪声
            noise_clip (float):    噪声限幅
            policy_freq (int):     延迟更新频率
            device (any):          训练设备

        Returns:
            None
        '''
        self.action_dim = action_dim

        self.actor1 = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor1 = copy.deepcopy(self.actor1)
        self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=actor_lr)

        self.actor2 = PolicyNet(state_dim, hidden_dim, action_dim, action_bound).to(device)
        self.target_actor2 = copy.deepcopy(self.actor2)
        self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=actor_lr)

        self.critic1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)

        self.critic2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_bound = action_bound
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
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
        state = torch.tensor([state], dtype=torch.float).to(self.device)

        action1 = self.actor1(state)
        action2 = self.actor2(state)

        q1 = self.critic1(state, action1)
        q2 = self.critic2(state, action2)

        action = action1 if q1 >= q2 else action2

        return action.cpu().data.numpy().flatten()

    def soft_update(self, net, target_net):
        '''
        软更新策略，
        采用当前网络参数和一部分过去网络参数一起更新，使得网络参数更加平滑

        Args:
            net (any):  更新网络
            target_net (any): 目标更新网络

        Returns:
            None
        '''
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self, transition_dict, batch_size=opt.batch_size):
        # cross-update scheme
        self.update(transition_dict, True, batch_size=batch_size)
        self.update(transition_dict, False, batch_size=batch_size)

    def update(self, transition_dict, update_a1=True, batch_size=100):
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

        next_action1 = self.target_actor1(next_states)
        next_action2 = self.target_actor2(next_states)

        noise = (
                torch.randn_like(actions) * self.policy_noise
        ).clamp(-self.noise_clip, self.noise_clip)

        next_action1 = (next_action1 + noise).clamp(-self.action_bound, self.action_bound)
        next_action2 = (next_action2 + noise).clamp(-self.action_bound, self.action_bound)

        target_Q1_a1 = self.target_critic1(next_states, next_action1)
        target_Q2_a1 = self.target_critic2(next_states, next_action2)

        target_Q1_a2 = self.target_critic1(next_states, next_action1)
        target_Q2_a2 = self.target_critic2(next_states, next_action2)

        target_Q1 = torch.min(target_Q1_a1, target_Q2_a1)
        target_Q2 = torch.min(target_Q1_a2, target_Q2_a2)

        target_Q = torch.max(target_Q1, target_Q2)
        target_Q = rewards + (1 - dones) * self.gamma * target_Q

        if update_a1:
            # Get current Q estimates
            current_Q1 = self.critic1(states, actions)
            # Compute critic loss
            critic1_loss = F.mse_loss(current_Q1, target_Q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            # 策略网络就是为了使Q值最大化
            actor1_loss = -torch.mean(self.critic1(states, self.actor1(states)))

            self.actor1_optimizer.zero_grad()
            actor1_loss.backward()
            self.actor1_optimizer.step()

            self.soft_update(self.actor1, self.target_actor1)  # 软更新策略网络
            self.soft_update(self.critic1, self.target_critic1)  # 软更新价值网络
        else:
            # Get current Q estimates
            current_Q2 = self.critic2(states, actions)

            # Compute critic loss
            critic2_loss = F.mse_loss(current_Q2, target_Q)

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            # 策略网络就是为了使Q值最大化
            actor2_loss = -torch.mean(self.critic2(states, self.actor2(states)))

            self.actor2_optimizer.zero_grad()
            actor2_loss.backward()
            self.actor2_optimizer.step()

            self.soft_update(self.actor2, self.target_actor2)  # 软更新策略网络
            self.soft_update(self.critic2, self.target_critic2)  # 软更新价值网络
        # return critic1_loss.cpu().detach().numpy()

    def save(self, filename):
        torch.save(self.critic1.state_dict(), filename + "_critic1.pt")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pt")
        torch.save(self.actor1.state_dict(), filename + "_actor1.pt")
        # torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pt")

        torch.save(self.critic2.state_dict(), filename + "_critic2.pt")
        # torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer.pt")
        torch.save(self.actor2.state_dict(), filename + "_actor2.pt")
        # torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer.pt")

    def load(self, filename):
        self.critic1.load_state_dict(torch.load(filename + "_critic1.pt"))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pt"))
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.actor1.load_state_dict(torch.load(filename + "_actor1.pt"))
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))
        self.target_actor1 = copy.deepcopy(self.actor1)

        self.critic2.load_state_dict(torch.load(filename + "_critic2.pt"))
        # self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer.pt"))
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.actor2.load_state_dict(torch.load(filename + "_actor2.pt"))
        # self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer.pt"))
        self.target_actor2 = copy.deepcopy(self.actor2)