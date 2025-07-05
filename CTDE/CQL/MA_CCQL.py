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

config = get_config()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def prep_dataloader(state_dim,dataset,batch_size=256):
    num_UAVs = 2
    state_start = str(0)
    state_end = str(state_dim - 1)
    action_start = str(state_dim)
    action_end = str(state_dim + num_UAVs - 1) # 1 --> (num_UAVs - 1)
    reward_col = str(state_dim + num_UAVs -1+1) # 2 --> (num_UAVs -1+1)
    next_state_start = str(state_dim + num_UAVs -1+2) # 3 --> (num_UAVs -1+2)
    next_state_end = str(state_dim + num_UAVs -1+2 + state_dim - 1)
    done_col = str(state_dim + num_UAVs -1+2 + state_dim - 1 + 1)    
    
    
    states_df = dataset.loc[:, state_start : state_end]
    actions_df = dataset.loc[:, action_start:action_end]
    rewards_df = dataset.loc[:,reward_col]
    next_states_df = dataset.loc[:, next_state_start : next_state_end]
    done_df = dataset.loc[:,done_col]

    tensors = {}
    tensors["observations"] = torch.tensor(states_df.values,dtype=torch.float)
    tensors["actions"] = torch.tensor(actions_df.values,dtype=torch.long)
    tensors["rewards"] = torch.tensor(rewards_df.values,dtype=torch.float).unsqueeze(1)
    tensors["next_observations"] = torch.tensor(next_states_df.values,dtype=torch.float)
    tensors["terminals"] = torch.tensor(done_df.values,dtype=torch.float).unsqueeze(1)
    


    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"],
                               tensors["next_observations"],
                               tensors["terminals"])
    
    
    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)

    return dataloader



class meta_Net_DNN(nn.Module):
    def __init__(self, state_size, action_size): # it only gets paramters from other network's parameters
        super(meta_Net_DNN, self).__init__()
#         self.input_shape = state_size
#         self.action_size = action_size
        self.activ = nn.ReLU()


    def forward(self, x, var):
        x = F.linear(x, var[0].to(device),var[1].to(device))
        x = torch.relu(x)
        x = F.linear(x, var[2].to(device),var[3].to(device))
        x = torch.relu(x)
        return F.linear(x, var[4].to(device),var[5].to(device))



def meta_dnn(state_size, action_size):
    net = meta_Net_DNN(state_size, action_size)
    return net




class basic_DNN(nn.Module):
    def __init__(self, state_size, action_size):
        super(basic_DNN, self).__init__()
        self.fc1 = nn.Linear(state_size[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def dnn(state_size, action_size):
    net = basic_DNN(state_size, action_size)
    return net



def cql_loss_cen(q_values, q_a):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
#         q_a = q_values.gather(1, current_action)
    
        return (logsumexp - q_a).mean()
    

def loss_update_cent(q1_loss, Net_params, Tar_params,optimizer):
    optimizer.zero_grad()
    q1_loss.backward()
    clip_grad_norm_(Net_params, 1.)
    optimizer.step()

    # ------------------- update target network ------------------- #
    soft_update_cent(Net_params, Tar_params, tau=1e-3)
    return q1_loss.detach().item()

def soft_update_cent(Net_params, Tar_params, tau):
    for target_param, local_param in zip(Tar_params, Net_params):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
def get_action_evaluate(env, state, epsilon, network):
    network.to(device)
    if random.random() > epsilon:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        network.eval()
        with torch.no_grad():
            action_values = network(state)
        network.train()
        action = np.argmax(action_values.cpu().data.numpy(), axis=1)
    else:
        action = random.choices(np.arange(env.action_space.shape[0]), k=1)
    return action


def soft_update(local_model, target_model):
    tau = 1e-3
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    return target_model



def multi_task_learning(config, env_test, Tasks_num, Tasks_test_num, net, target_net, dataloader, dataloader_meta_test, dataloader_test, device):
    
    Epoch_Reward = []
    Epoch_Reward_MAML = []
    First_loss_Epoch = []
    Second_Loss_Epoch = []
    agent_meta = []
    meta_optimiser = []

    for u in range(config.U):
        meta_optimiser_u = torch.optim.Adam(net[u].parameters(), 1e-4)
        meta_optimiser.append(meta_optimiser_u)
        
        agent_u = CQLAgent_meta(state_size=env_test.observation_space.shape,
                 action_size=env_test.action_space.shape[0],
                 device=device)
        agent_meta.append(agent_u)
        
    
    for epochs in range(500):
        first_loss = 0
        second_loss = 0
        iter_in_sampled_device = 0  # for averaging meta-devices
        
        task_list = np.random.choice(range(10), Tasks_num, replace=False)
        
        for t_cntt in range(Tasks_num):
            t = task_list[t_cntt]
            current_dataloader = dataloader[t]
            current_dataloader_meta_test = dataloader_meta_test[t]
            iter_in_sampled_device, first_loss_curr, second_loss_curr = maml(iter_in_sampled_device, env_test, agent_meta,
                                                                                 net, current_dataloader, current_dataloader_meta_test)
            first_loss = first_loss + first_loss_curr
            second_loss = second_loss + second_loss_curr
            
            first_loss = first_loss / Tasks_num
            second_loss = second_loss / Tasks_num
        
        for u in range(config.U):
            meta_optimiser[u].zero_grad()
            for f in net[u].parameters():
                f.grad = f.total_grad.clone() / Tasks_num
            meta_optimiser[u].step()  # Adam
        
            target_net[u] = soft_update(net[u], target_net[u])
            
            u_str = str(u)
            
            torch.save(net[u].state_dict(), 'net_'+u_str+'.pth')
            torch.save(target_net[u].state_dict(), 'target_net_'+u_str+'.pth')
        
        _ , Episode_Reward_MAML = evaluate_MAML(env_test, 50, dataloader_test,Tasks_test_num,True)
#         Episode_Reward = evaluate(config.episodes)
        Epoch_Reward_MAML.append(Episode_Reward_MAML)
        First_loss_Epoch.append(first_loss)
        Second_Loss_Epoch.append(second_loss)
#         Epoch_Reward.append(Episode_Reward[config.episodes-1])
        print("Epoch: {} | First loss: {} | Second loss: {} | Reward_MAML: {}".format(epochs+1, first_loss, second_loss, Episode_Reward_MAML,))

#         print("Epoch: {} | First loss: {} | Second loss: {} | Reward_MAML: {} | Reward: {}".format(epochs+1, first_loss, second_loss, Episode_Reward_MAML[config.episodes-1],Episode_Reward[config.episodes-1],))
        
    return net, Epoch_Reward_MAML, First_loss_Epoch, Second_Loss_Epoch

def maml(iter_in_sampled_device, env_test, agent_meta, net, current_dataloader, current_dataloader_meta_test):
    state_indexing = np.array([[0,1,4,5,6,7,8,9,10,11,12,13],
                               [2,3,4,5,6,7,8,9,10,11,12,13]])
    para_list_from_net = []
    net_meta_intermediate = []
    net_meta_intermediate_target = []
    for u in range(config.U): 
        net[u].zero_grad()
        para_list_from_net_u = list(map(lambda p: p[0], zip(net[u].parameters())))
        para_list_from_net.append(para_list_from_net_u)

        net_meta_intermediate_u = meta_dnn(env_test.observation_space.shape, env_test.action_space.shape[0])
        net_meta_intermediate.append(net_meta_intermediate_u)
        
        net_meta_intermediate_target_u = meta_dnn(env_test.observation_space.shape, env_test.action_space.shape[0])
        net_meta_intermediate_target.append(net_meta_intermediate_target_u)
        
    
    for inner_loop in range(1):
        if inner_loop == 0:
            for batch_idx, experience in enumerate(current_dataloader):
                states, actions, rewards, next_states, dones = experience
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                
                
                grad = []
                intermediate_updated_para_list = []                
                for u in range(config.U):
                    loss_CQL, Q_exp, Q_tar = agent_meta[u].learn_cql_cent((states[:,state_indexing[u]], actions[:,[u]], rewards, next_states[:,state_indexing[u]], dones),net_meta_intermediate[u],net_meta_intermediate_target[u],para_list_from_net[u])
                    
                    if(u==0):
                        loss_CQL_All = loss_CQL
                        Q_exp_All = Q_exp
                        Q_tar_All = Q_tar
                    else:
                        loss_CQL_All = loss_CQL_All + loss_CQL
                        Q_exp_All = Q_exp_All + Q_exp
                        Q_tar_All = Q_tar_All + Q_tar
                    
                loss_tensor = agent_meta[u].loss_calc_cent(loss_CQL_All,Q_exp_All,Q_tar_All)
#                 Net_params = list(agent[0].network.parameters()) + list(agent[1].network.parameters())
#                 Tar_params = list(agent[0].target_net.parameters()) + list(agent[1].target_net.parameters())

                loss = loss_tensor.detach().item()
#                 loss = loss_update_cent(loss_tensor,Net_params,Tar_params,optimizer)
                    
#                     loss[u], cql_loss, bellmann_error = agent_meta[u].learn_cql((states[:,state_indexing[u]], actions[:,[u]], rewards, next_states[:,state_indexing[u]], dones),net_meta_intermediate[u],net_meta_intermediate_target[u],para_list_from_net[u])
#                 loss, cql_loss, bellmann_error = agent_meta.learn_cql((states, actions, rewards, next_states, dones),net_meta_intermediate,net_meta_intermediate_target,para_list_from_net)
                
                first_loss_curr = float(loss_tensor)
        
                for u in range(config.U):
                    grad_u = torch.autograd.grad(loss_tensor, para_list_from_net[u], create_graph=True)
                    grad.append(grad_u)

                    intermediate_updated_para_list_u = list(map(lambda p: p[1] - 1e-3 * p[0], zip(grad[u], para_list_from_net[u])))
                    intermediate_updated_para_list.append(intermediate_updated_para_list_u)

                    net_meta_intermediate_target[u] = soft_update(net_meta_intermediate[u], net_meta_intermediate_target[u])
                    
        else:
            
            for batch_idx, experience in enumerate(current_dataloader):
                states, actions, rewards, next_states, dones = experience
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                
                intermediate_updated_para_list_u = []
                for u in range(config.U):
                    loss_CQL, Q_exp, Q_tar = agent_meta[u].learn_cql_cent((states[:,state_indexing[u]], actions[:,[u]], rewards, next_states[:,state_indexing[u]], dones),net_meta_intermediate[u],net_meta_intermediate_target[u],para_list_from_net[u])
                    
                    if(u==0):
                        loss_CQL_All = loss_CQL
                        Q_exp_All = Q_exp
                        Q_tar_All = Q_tar
                    else:
                        loss_CQL_All = loss_CQL_All + loss_CQL
                        Q_exp_All = Q_exp_All + Q_exp
                        Q_tar_All = Q_tar_All + Q_tar
                
                loss_tensor = agent_meta[u].loss_calc_cent(loss_CQL_All,Q_exp_All,Q_tar_All)
                loss = loss_tensor.detach().item()
                
#                 first_loss_curr = float(loss_tensor)
                
                
                for u in range(config.U):            
                    grad[u] = torch.autograd.grad(loss_tensor, intermediate_updated_para_list[u], create_graph=True)
                    intermediate_updated_para_list[u] = list(map(lambda p: p[1] - 1e-3 * p[0], zip(grad[u], intermediate_updated_para_list[u])))
                    net_meta_intermediate_target[u] = soft_update(net_meta_intermediate[u], net_meta_intermediate_target[u])

            ###########
    #### meta-update
    for batch_idx, experience in enumerate(current_dataloader_meta_test):
        states, actions, rewards, next_states, dones = experience
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        
        para_list_grad = []
        
        for u in range(config.U):
            loss_CQL, Q_exp, Q_tar = agent_meta[u].learn_cql_cent((states[:,state_indexing[u]], actions[:,[u]], rewards, next_states[:,state_indexing[u]], dones),net_meta_intermediate[u],net_meta_intermediate_target[u],para_list_from_net[u])

            if(u==0):
                loss_CQL_All = loss_CQL
                Q_exp_All = Q_exp
                Q_tar_All = Q_tar
            else:
                loss_CQL_All = loss_CQL_All + loss_CQL
                Q_exp_All = Q_exp_All + Q_exp
                Q_tar_All = Q_tar_All + Q_tar

        loss_tensor = agent_meta[u].loss_calc_cent(loss_CQL_All,Q_exp_All,Q_tar_All)
        
        second_loss_curr = float(loss_tensor)
        
        for u in range(config.U):
            para_list_grad_u = torch.autograd.grad(loss_tensor, para_list_from_net[u], create_graph=True)
            para_list_grad.append(para_list_grad_u)
            
            ind_f_para_list = [0] * config.U
            for f in net[u].parameters():
                if iter_in_sampled_device == 0:
                    f.total_grad = para_list_grad[u][ind_f_para_list[u]].data.clone()
                else:
                    f.total_grad = f.total_grad + para_list_grad[u][ind_f_para_list[u]].data.clone()
                ind_f_para_list[u] += 1
            

            net_meta_intermediate_target[u] = soft_update(net_meta_intermediate[u], net_meta_intermediate_target[u])
            
        iter_in_sampled_device = iter_in_sampled_device + 1
        
    return iter_in_sampled_device, first_loss_curr, second_loss_curr




def evaluate_MAML(env_test,epoch_number,dataloader_test,Tasks_test_num,if_MAML=False):
    Episode_Reward_ALL = []
    Final_Reward = 0
    state_indexing = np.array([[0,1,4,5,6,7,8,9,10,11,12,13],
                               [2,3,4,5,6,7,8,9,10,11,12,13]])
    
    for t in range(Tasks_test_num):
        net_evaluate = []
        target_net_evaluate = []
        optimizer_evaluate = []
        
        Dev_Coord = Dev_cord_Testing[t]
        i = I_Testing[t]
        j = J_Testing[t]
        
        Risky_region = np.array([[i,j],[i,j+1],[i,j+2],[i,j+3],[i,j+4],
                             [i+1,j],[i+1,j+1],[i+1,j+2],[i+1,j+3],[i+1,j+4],
                             [i+2,j],[i+2,j+1],[i+2,j+2],[i+2,j+3],[i+2,j+4],
                             [i+3,j],[i+3,j+1],[i+3,j+2],[i+3,j+3],[i+3,j+4]])
        
        k = K_Testing[t]
        DELTA = All_Deltas[k]
        
        env_test = environment(Dev_Coord,Risky_region,config,DELTA)    
        for u in range(config.U):
            u_str = str(u)

            net_evaluate_u = dnn(env_test.observation_space.shape, env_test.action_space.shape[0]).to(device)
            target_net_evaluate_u = dnn(env_test.observation_space.shape, env_test.action_space.shape[0]).to(device)

            net_evaluate_u.load_state_dict(torch.load('net_'+u_str+'.pth'))
            target_net_evaluate_u.load_state_dict(torch.load('target_net_'+u_str+'.pth'))

            net_evaluate.append(net_evaluate_u)
            target_net_evaluate.append(target_net_evaluate_u)


        Episode_Reward = []

        for i in range(1, epoch_number+1):
    #         loss = [0] * config.U
            for batch_idx, experience in enumerate(dataloader_test[t]):
                states, actions, rewards, next_states, dones = experience
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)

                for u in range(config.U):
                    with torch.no_grad():

                        Q_targets_next = target_net_evaluate[u](next_states[:,state_indexing[u]]).detach().max(1)[0].unsqueeze(1)
                        Q_targets = rewards/2 + (0.99 * Q_targets_next * (1 - dones))

                    Q_a_s = net_evaluate[u](states[:,state_indexing[u]])

                    Q_expected = Q_a_s.gather(1, actions[:,[u]])

                    cql1_loss = cql_loss_cen(Q_a_s, actions[:,[u]])

                    if(u==0):
                        loss_CQL_All = cql1_loss
                        Q_exp_All = Q_expected
                        Q_tar_All = Q_targets
                    else:
                        loss_CQL_All = loss_CQL_All + cql1_loss
                        Q_exp_All = Q_exp_All + Q_expected
                        Q_tar_All = Q_tar_All + Q_targets

                bellman_error = F.mse_loss(Q_exp_All, Q_tar_All)

                q1_loss = 1*loss_CQL_All + 0.5 * bellman_error # alpha = 1


                Net_params = list(net_evaluate[0].parameters()) + list(net_evaluate[1].parameters())
                Tar_params = list(target_net_evaluate[0].parameters()) + list(target_net_evaluate[1].parameters())

                optimizer_evaluate = torch.optim.Adam(params=Net_params, lr=1e-4)
                loss = loss_update_cent(q1_loss,Net_params,Tar_params,optimizer_evaluate)

            if i % config.eval_every == 0:
                eval_reward = evaluate(env_test, net_evaluate)

                Episode_Reward.append(eval_reward)
            
        Episode_Reward_ALL.append(Episode_Reward)
        Final_Reward = Final_Reward + Episode_Reward[-1]
        
    return Episode_Reward_ALL, Final_Reward/Tasks_test_num



def evaluate(env, network, eval_runs=50): 
    """
    Makes an evaluation run with the current policy
    """
    state_indexing = np.array([[0,1,4,5,6,7,8,9,10,11,12,13],
                               [2,3,4,5,6,7,8,9,10,11,12,13]])
    reward_batch = []
    for ii in range(eval_runs):
        state = env.reset()

        rewards = 0
        action = [0] * env.U
        while True:
            for u in range(config.U): 
                action_u = get_action_evaluate(env, state[state_indexing[u]], 0, network[u])
                action[u] = action_u[0]

            next_state, reward, done = env.step(state,action)
    
            rewards += reward
            
            state = next_state
            if done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)








