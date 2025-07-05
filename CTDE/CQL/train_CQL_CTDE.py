import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import numpy as np
from collections import deque
import torch
import argparse
from agent_meta import CQLAgent_meta
from MA_CCQL import multi_task_learning, dnn, prep_dataloader
import glob
from utils import collect_random, get_config
import random
from Environment import environment
from torch.utils.data import DataLoader, TensorDataset


import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import math
import copy
import random
import itertools
import torch
from collections import deque
import pandas as pd



## Define constants for the dataset collection relative to the online training
# Meta_Train

All_Deltas = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

I_Training = [1, 2, 6, 4, 5, 6, 1, 2, 4, 0]
J_Training = [4, 1, 3, 3, 2, 1, 1, 1, 1, 2]
K_Training = [5, 9, 9, 4, 4, 4, 7, 1, 2, 2]

Dev_cord_Training = [np.array([[7, 6],[5, 1],[4, 0],[1, 6],[5, 7],[1, 6],[7, 9],[1, 0],[2, 0],[8, 1]]), 
                     np.array([[5, 1],[2, 5],[3, 3],[3, 2],[8, 1],[9, 6],[4, 3],[4, 8],[3, 8],[7, 1]]), 
                     np.array([[2, 8],[4, 2],[5, 1],[2, 5],[3, 3],[3, 2],[8, 1],[9, 6],[4, 3],[4, 8]]), 
                     np.array([[5, 2],[4, 0],[8, 3],[7, 0],[3, 5],[5, 8],[2, 1],[9, 3],[1, 2],[8, 9]]), 
                     np.array([[5, 1],[2, 5],[3, 3],[3, 2],[8, 1],[9, 6],[4, 3],[4, 8],[3, 8],[7, 1]]), 
                     np.array([[5, 4],[7, 6],[4, 2],[9, 3],[2, 8],[4, 2],[5, 1],[2, 5],[3, 3],[3, 2]]), 
                     np.array([[3, 4],[2, 0],[5, 8],[7, 8],[8, 4],[6, 3],[4, 5],[1, 4],[5, 2],[4, 0]]), 
                     np.array([[7, 1],[3, 4],[0, 2],[3, 4],[2, 0],[5, 8],[7, 8],[8, 4],[6, 3],[4, 5]]), 
                     np.array([[2, 8],[4, 2],[5, 1],[2, 5],[3, 3],[3, 2],[8, 1],[9, 6],[4, 3],[4, 8]]), 
                     np.array([[4, 3],[4, 8],[3, 8],[7, 1],[3, 4],[0, 2],[3, 4],[2, 0],[5, 8],[7, 8]])]


I_Testing = [3, 3, 6, 2, 0]
J_Testing = [1, 0, 0, 0, 4]
K_Testing = [3, 6, 8, 9, 9]


Dev_cord_Testing = [np.array([[3, 4],[2, 0],[5, 8],[7, 8],[8, 4],[6, 3],[4, 5],[1, 4],[5, 2],[4, 0]]),
                    np.array([[7, 0],[3, 5],[5, 8],[2, 1],[9, 3],[1, 2],[8, 9],[9, 9],[0, 1],[0, 3]]), 
                    np.array([[5, 8],[2, 1],[9, 3],[1, 2],[8, 9],[9, 9],[0, 1],[0, 3],[1, 0],[6, 3]]), 
                    np.array([[3, 8],[7, 1],[3, 4],[0, 2],[3, 4],[2, 0],[5, 8],[7, 8],[8, 4],[6, 3]]), 
                    np.array([[2, 1],[9, 3],[1, 2],[8, 9],[9, 9],[0, 1],[0, 3],[1, 0],[6, 3],[5, 5]])]


# Meta_Test

Dev_Coord = np.array([[3,1],[7,2],[6,7],[1,6],[7,5],[8,5],[9,1],[6,1],[4,7],[2,3]])

Risky_region = np.array([[3,2],[3,3],[3,4],[3,5],[3,6],
                         [4,2],[4,3],[4,4],[4,5],[4,6],
                         [5,2],[5,3],[5,4],[5,5],[5,6],
                         [6,2],[6,3],[6,4],[6,5],[6,6]])


DELTA = 350
config = get_config()
env_test = environment(Dev_Coord,Risky_region,config,DELTA)

Tasks_num = 5
Tasks_test_num = 1



## Agents and Datasets
np.random.seed(config.seed)
random.seed(config.seed)
torch.manual_seed(config.seed)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = []
target_net = []

for u in range(config.U):
        net_u = dnn(env_test.observation_space.shape, env_test.action_space.shape[0])
        target_net_u = dnn(env_test.observation_space.shape, env_test.action_space.shape[0])

        net.append(net_u)
        target_net.append(target_net_u)
        

dataloader_meta_train = []
dataloader_meta_test = []
for t in range(1,11):
    indx_str = str(t)
    dataset = pd.read_csv(r"/homedir01/Meta_Offline_MARL/Datasets/Dataset_"+indx_str+'.csv')
    dataloader_t = prep_dataloader(state_dim = 14, batch_size=64, dataset = dataset)
    dataloader_meta_train.append(dataloader_t)
    
    dataset = pd.read_csv(r"/homedir01/Meta_Offline_MARL/Datasets/Dataset_test_"+indx_str+'.csv')
    dataloader_t = prep_dataloader(state_dim = 14, batch_size=64, dataset = dataset)
    dataloader_meta_test.append(dataloader_t)
    


dataloader_train = []
dataloader_test = []
for t in range(11,16):
    indx_str = str(t)
    dataset = pd.read_csv(r"/homedir01/Meta_Offline_MARL/Datasets/Dataset_"+indx_str+'.csv')
    dataloader_t = prep_dataloader(state_dim = 14, batch_size=64, dataset = dataset)
    dataloader_train.append(dataloader_t)
    

    
    
    
def train(config, env_test, Tasks_num, Tasks_test_num, net, target_net, dataloader_meta_train, dataloader_meta_test, dataloader_train, device):
    net,Episode_Reward,First_loss_Epoch, Second_Loss_Epoch = multi_task_learning(config, env_test, Tasks_num, Tasks_test_num, net, target_net, dataloader_meta_train, dataloader_meta_test, dataloader_train, device)


    

if __name__ == "__main__":
    train(config, env_test, Tasks_num, Tasks_test_num, net, target_net, dataloader_meta_train, dataloader_meta_test, dataloader_train, device)













