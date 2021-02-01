import torch
from torch import nn, optim
import numpy as np
import torch.nn.functional as F
from collections import deque
import random

# from rlmodel.MADDPG.param import *
from param import *


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.input_size, self.output_size = args
        self.actor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )

    def forward(self, inputs):
        policy_op = self.actor(inputs)
        return policy_op

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.input_size, self.output_size = args
        self.critic = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, inputs):
        critic_op = self.critic(inputs)
        return critic_op

class Replaybuffer(object):
    def __init__(self, args):
        self.mem_len, self.device = args
        self.buffer = deque(maxlen = self.mem_len)

    def save_trans(self, trans):
        self.buffer.append(trans)

    def sample_batch(self, batch_size = 32):
        s_ls, obs_ls, a_ls, r_ls, s_next_ls, obs_next_ls, done_ls = ([] for _ in range(7))
        trans_batch = random.sample(self.buffer, batch_size)
        for trans in trans_batch:
            s, obs, a, r, s_next, obs_next, done = trans
            s_ls.append(s)
            obs_ls.append(obs)
            a_ls.append(a)
            r_ls.append([r])
            s_next_ls.append(s_next)
            obs_next_ls.append(obs_next)
            done_ls.append([done])
        return torch.FloatTensor(s_ls).to(self.device),\
            torch.FloatTensor(obs_ls).to(self.device),\
            np.array(a_ls),\
            torch.FloatTensor(r_ls).to(self.device),\
            torch.FloatTensor(s_next_ls).to(self.device),\
            torch.FloatTensor(obs_next_ls).to(self.device),\
            torch.FloatTensor(done_ls).to(self.device)


class DDPG(nn.Module):
    def __init__(self, args):
        super(DDPG, self).__init__()
        self.id, self.state_size, self.obs_size, self.joint_action_size, self.action_size, self.lr, self.device = args
        self.critic_net = Critic(args = (self.state_size + self.joint_action_size, self.action_size))
        self.actor_net = Actor(args = (self.obs_size, self.action_size))
        self.buffer = Replaybuffer(args = (MEM_SIZE, self.device))
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr = self.lr)
        self.optimizer_actor = optim.Adam(self.actor_net.parameters(), lr = self.lr)

    def get_critic(self, inputs):
        return self.critic_net(inputs)

    def get_policy(self, inputs):
        return self.actor_net(inputs)

    def get_action(self, inputs, vec = False):
        policy_op = self.get_policy(inputs)
        noise = torch.rand_like(policy_op)
        action = F.softmax(policy_op - torch.log(-torch.log(noise)), -1)        # 限制范围  合法动作
        return action.data.numpy() if not vec else action

    
class MADDPG(nn.Module):
    def __init__(self, args):
        super(MADDPG, self).__init__()
        self.state_size, self.obs_size, self.joint_action_size, self.action_size, self.n_agents, self.lr, self.device = args
        self.agent_models = [DDPG(args = (i, self.state_size, self.obs_size[i], self.joint_action_size, self.action_size[i], self.lr, self.device)) for i in range(self.n_agents)]
        self.target_models = [DDPG(args = (i, self.state_size, self.obs_size[i], self.joint_action_size, self.action_size[i], self.lr, self.device)) for i in range(self.n_agents)]
        self.copy_weight()

    def copy_weight(self):
        for agent_model, target_model in zip(self.agent_models, self.target_models):
            target_model.load_state_dict(agent_model.state_dict())

    def soft_update_model(self):
        for agent_model, target_model in zip(self.agent_models, self.target_models):
            for raw_param, target_param in zip(agent_model.parameters(), target_model.parameters()):
                target_param = TOI * raw_param + (1 - TOI) * target_param

    def get_action(self, inputs):
        action_ls = []
        for agent_i in range(self.n_agents):
            action_ls.append(self.agent_models[agent_i].get_action(torch.FloatTensor(inputs[agent_i]).to(self.device)))
        return action_ls

    def train(self, gamma = GAMMA, batch_size = BATCH_SIZE):
        # 各个更新
        for agent_i in range(self.n_agents):
            s, obs, a_ls, r, s_next, obs_next, done = self.agent_models[agent_i].buffer.sample_batch(batch_size)
            q_val = self.agent_models[agent_i].get_critic(torch.cat([s, torch.FloatTensor(np.concatenate([a_ls[:,i_a,:] for i_a in range(self.n_agents)], -1)).to(self.device)], -1))
            
            a_next_ls = []
            cur_idx = 0
            for i in range(self.n_agents):
                a_next_ls.append(torch.FloatTensor(self.agent_models[i].get_action(s[:, cur_idx:cur_idx + self.agent_models[i].actor_net.input_size])).to(self.device))
                cur_idx += self.agent_models[i].actor_net.input_size
            q_target = r + gamma * self.target_models[agent_i].get_critic(torch.cat([s, torch.cat(a_next_ls, -1)], -1)) * (1 - done)
            
            critic_loss = ((q_target.detach() - q_val) ** 2).mean()
            self.agent_models[agent_i].optimizer_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.agent_models[agent_i].critic_net.parameters(), 10)
            self.agent_models[agent_i].optimizer_critic.step()

            a_vec = self.agent_models[agent_i].get_action(obs, vec = True)
            a = torch.FloatTensor(a_ls).to(self.device)
            a[:,agent_i,:] = a_vec
            a = a.view(batch_size, -1)
            policy_loss = -(self.agent_models[agent_i].get_critic(torch.cat([s, a], -1))).mean()
            self.agent_models[agent_i].optimizer_actor.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.agent_models[agent_i].actor_net.parameters(), 10)
            self.agent_models[agent_i].optimizer_actor.step()

        self.soft_update_model()

    

            





