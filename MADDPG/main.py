import torch
from torch import nn, optim
import torch.nn.functional as F

from run import *
from param import *
from utils import *
from make_env import make_env

def main(scen_name, train_mode):
    env_raw = make_env(scenario_name = scen_name)
    env = Env_interface(args = (env_raw, scen_name))
    run(env, train_mode)

if __name__ == "__main__":
    main(scen_name = 'simple_tag', train_mode = 'display')