#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        SlideBars.py
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

# here put the import lib

import pybullet as p
import numpy as np
import torch
# from TD3.core import combined_shape

class SlideBars():
    def __init__(self, Id):
        self.Id = Id
        self.motorNames = []
        self.motorIndices = []
        self.motorLowerLimits = []
        self.motorUpperLimits = []
        self.slideIds = []

        self.numJoints = p.getNumJoints(self.Id)

    def add_slidebars(self):
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.Id, i)
            jointName = jointInfo[1].decode('ascii')
            qIndex = jointInfo[3]
            lowerLimits = jointInfo[8]
            upperLimits = jointInfo[9]
            if qIndex > -1:
                self.motorNames.append(jointName)
                self.motorIndices.append(i)
                self.motorLowerLimits.append(lowerLimits)
                self.motorUpperLimits.append(upperLimits)

        for i in range(len(self.motorIndices)):
            if self.motorLowerLimits[i] <= self.motorUpperLimits[i]:
                slideId = p.addUserDebugParameter(self.motorNames[i],
                                                  self.motorLowerLimits[i],
                                                  self.motorUpperLimits[i], 0)
            else:
                slideId = p.addUserDebugParameter(self.motorNames[i],
                                                  self.motorUpperLimits[i],
                                                  self.motorLowerLimits[i], 0)
            self.slideIds.append(slideId)

        return self.motorIndices

    def get_slidebars_values(self):
        slidesValues = []
        for i in self.slideIds:
            value = p.readUserDebugParameter(i)
            slidesValues.append(value)
        return slidesValues


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e4)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # statetemp=state_dim.flatten()

        self.state = np.zeros(combined_shape(max_size, state_dim),
                                dtype=np.float32)
        self.action = np.zeros(combined_shape(max_size, action_dim),
                                dtype=np.float32)
        self.next_state = np.zeros(combined_shape(max_size, state_dim),
                                dtype=np.float32)
        self.reward = np.zeros((max_size, 1),dtype=np.float32)
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
