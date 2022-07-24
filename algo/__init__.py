#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
 ------------------------------------------------------------------
 @File Name:        __init__.py
 @Created:          2022/7/21 9:36
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
  2022/7/21   | v1.0      | HHH            | Create file                     
 ------------------------------------------------------------------
'''
from .TD3.TD3_mlp import TD3_MLP
from .TD3.TD3_cnn import TD3_CNN
from .DDPG.DDPG_mlp import DDPG_MLP
from .DDPG.DDPG_cnn import DDPG_CNN
from .DARC.DARC_mlp import DARC_MLP
from .DARC.DARC_cnn import DARC_CNN
from .DADDPG.DADDPG_mlp import DADDPG_MLP
from .DADDPG.DADDPG_cnn import DADDPG_CNN
from .DATD3.DATD3_mlp import DATD3_MLP
from .DATD3.DATD3_cnn import DATD3_CNN
from .DQN.DQN_mlp import DQN_MLP
from .DQN.DQN_cnn import DQN_CNN
