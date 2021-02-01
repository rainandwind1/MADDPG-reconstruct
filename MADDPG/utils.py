import torch
from torch import nn, optim
import torch.nn.functional as F

from param import *
import numpy as np

class Env_interface(object):
    def __init__(self, args):
        self.env, self.name  = args
        self.obs_size_ls = []
        self.action_size_ls = []
        self.joint_action_size = 0
        self.state_size = 0
        self.n_agent = 0
        self.initilize()

    def get_state(self, obs_ls):
        state = np.concatenate(obs_ls, -1)
        return state

    def step(self, action_ls):
        obs_next_ls, reward_ls, done_ls, info_ls = self.env.step(action_ls)
        state_next = self.get_state(obs_next_ls)
        return state_next, obs_next_ls, reward_ls, done_ls, info_ls

    def reset(self):
        obs_ls = self.env.reset()        # 初始化状态
        return obs_ls
        
    def initilize(self):
        obs_ls = self.env.reset()
        self.n_agents = len(obs_ls)
        for cv in obs_ls:
            self.obs_size_ls.append(len(cv))
            self.state_size += len(cv)
        for action_space in self.env.action_space:
            self.action_size_ls.append(action_space.n)
            self.joint_action_size += action_space.n