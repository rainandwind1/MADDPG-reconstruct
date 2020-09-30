import torch
from torch import nn, optim
import torch.nn.functional as F
from make_env import make_env
import numpy as np
import random
import argparse
import os
from model import *

SCENAION_NAME = 'simple_tag'    # choose scenario for training
parser = argparse.ArgumentParser(description='Base init and setup for training or display')
parser.add_argument('-scen_name', type=str, help='Choose scenarios for training or display', default=SCENAION_NAME)



def single_cross(chro1, chro2):
    keys1 = list(chro1.state_dict())
    keys2 = list(chro2.state_dict())
    for key in keys1:
        if key not in keys2: continue
        
        W1 = chro1.state_dict()[key]
        W2 = chro2.state_dict()[key]

        # weights 
        if len(W1.shape) == 2:
            num_variables = W1.shape[0]
            # crossover operation [indexed by row]
            try: num_cross_overs = random.randint(0, int(len(num_variables) * 0.3))
            except: num_cross_overs = 1
            for i in range(num_cross_overs):
                receiver_choice = random.random()
                if receiver_choice < 0.5:
                    ind_cr = random.randint(0, W1.shape[0] - 1)
                    W1[ind_cr,:] = W2[ind_cr,:]
                else:
                    ind_cr = random.randint(0, W1.shape[0] - 1)
                    W1[ind_cr,:] = W2[ind_cr,:]
        
        # bias or LayerNorm
        elif len(W1.shape) == 1:
            if random.random() < 0.8: continue
            num_variables = W1.shape[0]
            for i in range(1):
                receiver_choice = random.random()
                if receiver_choice < 0.5:
                    ind_cr = random.randint(0, W1.shape[0] - 1)
                    W1[ind_cr] = W2[ind_cr]
                else:
                    ind_cr = random.randint(0, W1.shape[0] - 1)
                    W1[ind_cr] = W2[ind_cr]
    return chro1


# 环境交互
def Rollout(env, policy_team, target_team, critic, target_critic, replay_buffer_ls, generation_idx, team_idx, noise, kersi):
    global total_step
    global train_flag
    fitness = 0
    for j in range(1, kersi):
        obs_ls = env.reset()
        score_ls = np.array([0. for _ in range(env.n)]) # n个代理的回合得分表
        for step in range(DONE_INTERVAL):
            total_step += 1
            if RENDER_FLAG:
                env.render()
            
            # 动作序列
            action_ls = policy_team.get_action(obs_ls, noise)
            
            obs_next_ls, reward_ls, done_ls, info_ls = env.step(action_ls)
            score_ls += reward_ls
            for d in done_ls:
                if (total_step % 60 and total_step > 0) or d:
                    done_flag_ls = [0.] * env.n
                else:
                    done_flag_ls = [1.] * env.n
            
            # save transitions
            total_s = []
            total_s_next = []
            action_vec = []
            for t in range(len(obs_ls)):
                total_s += list(obs_ls[t])
                total_s_next += list(obs_next_ls[t])
                action_vec += list(action_ls[t])
            for n in range(len(replay_buffer_ls)):
                replay_buffer_ls[n].save_trans((obs_ls[n], total_s, action_vec, reward_ls[n], total_s_next, done_flag_ls[n]))
            
            obs_ls = obs_next_ls
            
            # train agent net
            if TRAIN_KEY:
                if total_step > 3000:
                    if total_step % 5 == 0:
                        train_flag = True
                        policy_team.train(target_team, critic, target_critic, replay_buffer_ls, GAMMA, BATCH_SIZE, TOI)
            
            # ******* 打印回合结果 ********
            if step == DONE_INTERVAL - 1:
                # if generation_idx == 0:
                #     score_mem = score_ls
                # else:
                #     score_ls = 0.99 * score_mem + 0.01 * score_ls
                #     score_mem = score_ls
                print("Epoch:{}  Team:{}".format(generation_idx + 1, team_idx))
                for idx, score in enumerate(score_ls):
                    print("agent{} score:{} train_flag:{}".format(idx, score, train_flag))
        fitness += sum(score_ls)
    return fitness / kersi



if __name__ == "__main__":

    param_path = '.\param'
    log_path = '.\info'
    if not os.path.exists(param_path):
        print("创建参数文件夹")
        os.makedirs(param_path)
    if not os.path.exists(log_path):
        print("创建日志文件夹")
        os.makedirs(log_path)

    # Hpyerparameters
    LEARNING_RATE = 1e-3
    GAMMA = 0.98
    BATCH_SIZE = 128
    DONE_INTERVAL = 100 
    SAVE_INTERVAL = 100
    MAX_EPOCH = 30000
    MAX_GENERATION = 100000
    MEM_LEN = 10000
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    ELITES_NUM = 3
    M = 8
    RENDER_FLAG = False
    AGENT_NUM = 4
    BUFFER_SIZE = 32
    MUTATION_PROB = 0.9
    MUTATION_FRACTION = 0.1
    MUTATION_STRENGTH = 0.1
    SUPER_MUTATION_PROB = 0.05
    RESET_MUTATION_PROB = 0.05
    TOI = 0.01

    args = parser.parse_args()
    env_name = args.scen_name
    display = False
    train_flag = False
    total_step = 0
    score_mem = 0

    if display:
        render_flag, LOAD_KEY, TRAIN_KEY = [True, True, False]
    else:
        render_flag, LOAD_KEY, TRAIN_KEY = [False, False, True]


    env = make_env(env_name)
    obs_ls = env.reset()        # 初始化状态
    global_input_size = 0
    for cv in obs_ls:
        global_input_size += len(cv)
    for action_space in env.action_space:
        global_input_size += action_space.n

    # 初始化模型
    model_ls = [MERL(args = [Policy(args = (len(obs_ls[idx]),  env.action_space[idx].n, idx, LEARNING_RATE, BUFFER_SIZE, DEVICE, idx)) for idx in range(len(env.world.agents))]).to(DEVICE) for _ in range(M)]    # 种群 size = M 
    target_model = [MERL(args = [Policy(args = (len(obs_ls[idx]),  env.action_space[idx].n, idx, LEARNING_RATE, BUFFER_SIZE, DEVICE, idx)) for idx in range(len(env.world.agents))]).to(DEVICE) for _ in range(M)]
    replay_buffer_ls = [ReplayBuffer(args = (MEM_LEN, DEVICE)) for _ in range(len(env.world.agents))]
    critic = Critic(args = (global_input_size, LEARNING_RATE, DEVICE)).to(DEVICE)                       # 共享一个 critic网络
    target_critic = Critic(args = (global_input_size, LEARNING_RATE, DEVICE)) .to(DEVICE)
    
    # load weights
    # if LOAD_KEY:
    #     continue

    # 权重拷贝
    for target, model in zip(target_model, model_ls):
        target.load_state_dict(model.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    # 进化算法的基本组成部分的初始化
    population_idls = [i for i in range(M)]
    population_ls = [model_ls[i] for i in population_idls]

    # main iteration running
    for g_i in range(MAX_GENERATION):
        fitness_ls = []
        for idx, policy_team in enumerate(population_ls):
            print("Rollout running...")
            g = Rollout(env, policy_team, target_model[idx], critic, target_critic, replay_buffer_ls, g_i, idx, noise = False, kersi = 2)       # g 就是该 team 的 fitness
            _ = Rollout(env, policy_team, target_model[idx], critic, target_critic, replay_buffer_ls, g_i, idx, noise = True, kersi = 1)        # noise: Gaussian noise
            fitness_ls.append((idx, g))
        fitness_ls = sorted(fitness_ls, key = lambda item: item[1])         # (11): rank the population pop_pi based on fitness scores
        elites_ls = [population_ls[elite[0]] for elite in fitness_ls[:ELITES_NUM]]         # elites list
        # forming tournament list
        tournament_ls = []
        while len(tournament_ls) < M - ELITES_NUM:
            # single-point crossover
            elite = random.sample(elites_ls, 1)[0]
            if not tournament_ls:
                tournament_ls.append(elite)
            else:
                tourn = random.sample(tournament_ls, 1)[0]
                new_member = single_cross(elite, tourn)
                tournament_ls.append(new_member)
        population_ls = elites_ls + tournament_ls

        # 保存网络权重等信息
        if (g_i+1) % SAVE_INTERVAL == 0:
            print('save process')
            best_model = model_ls[elites_ls[0]]
            for idx, model in enumerate(best_model):
                print('agent' + str(idx))
                torch.save(model.state_dict(), param_path + '/MERL_agent' + str(idx) + '_' + str(g_i + 1) + '.pkl')
    env.close()
