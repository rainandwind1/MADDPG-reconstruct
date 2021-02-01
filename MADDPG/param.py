import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 8e-4
GAMMA = 0.98
MAX_STEPS = 200
MAX_EPISODES = 10000000
BATCH_SIZE = 32
# PATH = 'rlmodel/MADDPG/param'
PATH = './param'
EXPLORATION_STEPS = 10000
MEM_SIZE = 50000
TOI = 0.01

ACTION_MAX = 360
ACTION_MIN = -360
ACTION_SIZE = 1
