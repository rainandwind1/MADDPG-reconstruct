import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

# from rlmodel.MADDPG.model import *
# from rlmodel.MADDPG.param import *
from model import *
from param import *
from utils import *

def run(env, train_display):

    obs_ls = env.reset()
    
    if train_display == 'display':
        LOAD_KEY, TRAIN_KEY, RENDER = True, False, True
    else:
        LOAD_KEY, TRAIN_KEY, RENDER = False, True, False

    train_flag = False
    total_step = 0
    score_pre = None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模型初始化
    model = MADDPG(args = (env.state_size, env.obs_size_ls, env.joint_action_size, env.action_size_ls, env.n_agents, LEARNING_RATE, device)).to(device)
    
    # 载入模型
    if LOAD_KEY:
        print("Load model ...")
        check_point = torch.load("MADDPG/param/MADDPG_epoch_800_simple_tag.pkl")
        model.load_state_dict(check_point)

    for ep_i in range(MAX_EPISODES):
        obs = env.reset()
        terminated = False
        state = env.get_state(obs)
        score = np.zeros(env.n_agents)
        for step_i in range(MAX_STEPS):
            total_step += 1
            action = model.get_action(obs)
            state_next, obs_next, reward, done, info = env.step(action)
            if RENDER:
                env.env.render()
            
            score += reward
            for agent_i in range(env.n_agents):
                model.agent_models[agent_i].buffer.save_trans((state, obs[agent_i], action, reward[agent_i], state_next, obs_next[agent_i], done[agent_i]))
            
            # 更新mem
            state = state_next
            obs = obs_next

            if TRAIN_KEY and total_step > 10000:
                train_flag = True
                model.train()

        if (ep_i+1) % 200 == 0:
            print("Save model ...")
            torch.save(model.state_dict(), PATH + "/MADDPG_epoch_" + str(ep_i+1) + '_{}'.format(env.name) + '.pkl')
            print("Ok")
        
        if ep_i > 1:
            score = 0.95 * score_pre + 0.05 * score 
        score_pre = score

        for agent_i in range(env.n_agents):
            print("Total reward in episode {} = {:.3},  training:  {}".format(ep_i, score[agent_i], train_flag))
        print('\n')
            
            

