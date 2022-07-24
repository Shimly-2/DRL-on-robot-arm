#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        net_mlp.py
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

class PolicyNet(torch.nn.Module):
    '''策略网络'''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x))))
        return torch.tanh(self.fc3(x)) * self.action_bound


class QValueNet(torch.nn.Module):
    '''估值网络'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        # Q1 architecture
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.fc4 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        cat = torch.cat([state, action], dim=1)  # 拼接状态和动作
        q1 = F.relu(self.fc2(F.relu(self.fc1(cat))))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc5(F.relu(self.fc4(cat))))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, state, action):
        cat = torch.cat([state, action], dim=1)  # 拼接状态和动作

        q1 = F.relu(self.fc2(F.relu(self.fc1(cat))))
        q1 = self.fc3(q1)
        return q1