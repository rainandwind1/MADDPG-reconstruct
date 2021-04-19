import collections
import torch
import random
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import copy

# replay buffer
class ReplayBuffer():
    def __init__(self, args):
        self.size, self.device = args
        self.buffer = collections.deque(maxlen = self.size)
    
    def save_trans(self, trans):
        self.buffer.append(trans)
    
    def sample_batch(self, batch_size):
        # (obs_ls[n], total_s, action_ls, reward_ls[n], total_s_next, done_flag_ls[n])
        p_obs, s_ls, a_ls, r_ls, s_next_ls, done_ls = ([] for i in range(6))
        trans_batch = random.sample(self.buffer, batch_size)
        for trans in trans_batch:
            obs, s, a, r, s_next, done = trans
            p_obs.append(obs)
            s_ls.append(s)
            a_ls.append(a)
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_ls.append([done])
        return torch.FloatTensor(p_obs).to(self.device),\
                torch.FloatTensor(s_ls).to(self.device),\
                torch.FloatTensor(a_ls).to(self.device),\
                torch.FloatTensor(r_ls).to(self.device),\
                torch.FloatTensor(s_next_ls).to(self.device),\
                torch.FloatTensor(done_ls).to(self.device)

# critic (Q value)
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.input_size, self.lr, self.device = args
        self.critic = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def get_critic(self, inputs):
        q_val = self.critic(inputs)
        return q_val

# actor policy op
class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.input_size, self.output_size, self.idx, self.lr, self.buffer_size, self.device, self.name = args
        self.actor = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )                               
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def get_action(self, inputs, epsilon = 1., batch = False, noise = True):
        if not batch:
            inputs = torch.FloatTensor(inputs).to(self.device)
        policy = self.actor(inputs)
        if noise:
            noise = torch.rand_like(policy) * epsilon
            return F.softmax(policy - torch.log(-torch.log(noise)), -1)
        else:
            return F.softmax(policy, -1)

    
    def train(self, target_policy_ls, critic, target_critic, replaybuffer, gamma, batch_size, toi):
        obs, s, a, r, s_next, done = replaybuffer.sample_batch(batch_size)
        critic_op = critic.get_critic(torch.cat([s, a], -1))
        # td target
        a_next = []
        pre, cur, a_pre,a_cur = 0, 0, 0, 0
        for idx, policy in enumerate(target_policy_ls.policy_team):
            cur += policy.input_size
            a_cur += policy.output_size
            if idx == self.name:
                mem_pre, mem_cur = a_pre, a_cur
            a_par = policy.get_action(s[:, pre:cur], batch = True)
            pre = cur
            a_pre = a_cur
            a_next.append(a_par.detach().cpu().clone().numpy())
        a_next = np.concatenate(a_next, -1)
        target_op = r + gamma * target_critic.get_critic(torch.cat([s_next, torch.FloatTensor(a_next).to(self.device)], -1)) * done
        td_error = target_op.detach() - critic_op
        # critic update
        critic_loss = (td_error ** 2).mean()
        critic.optimizer.zero_grad()
        critic_loss.backward()
        critic.optimizer.step()

        # actor update
        a_grad = a
        a_grad[:, mem_pre:mem_cur] = self.get_action(obs, batch = True)
        policy_loss = (- critic.get_critic(torch.cat([s, a_grad], -1))).mean() 
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        for model, target in zip(self.parameters(), target_policy_ls.policy_team[self.name].parameters()):
            target.data.copy_(toi * model.data + (1 - toi) * target.data)
        for target, model in zip(target_critic.parameters(), critic.parameters()):
            target.data.copy_(toi * model.data + (1 - toi) * target.data)

# Team Model MERL
class MERL(nn.Module):
    def __init__(self, args):
        super(MERL, self).__init__()
        self.model_ls = args
        self.policy_team = nn.ModuleList(self.model_ls)

    def get_action(self, obs_ls, noise):
        action_ls = []
        for i, model in enumerate(self.policy_team):
            action_vec_i = model.get_action(obs_ls[i], noise = noise) 
            action_ls.append(action_vec_i.detach().cpu().clone().numpy())
        return action_ls


    def train(self, target_team, critic, target_critic, replaybuffer, gamma, batch_size, toi):
        for i, model in enumerate(self.policy_team):
            model.train(target_team, critic, target_critic, replaybuffer[i], gamma, batch_size, toi)

