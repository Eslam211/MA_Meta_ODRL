import torch
import torch.nn as nn
# from networks import DDQN
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
import pdb


class CQRAgent_meta():
    def __init__(self, state_size, action_size, hidden_size=256, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.TAU = 1e-2
        self.GAMMA = 0.99
        self.BATCH_SIZE = 64
        self.Q_updates = 0
        self.n_step = 1
        self.N = 8
        self.quantile_tau = torch.FloatTensor([(2 * i + 1) / (2.0 * self.N) for i in range(0,self.N)]).to(device)
        #self.quantile_tau = torch.FloatTensor((2 * i + 1) / (2.0 * self.N)).view(1, -1).to(device)
        

#         self.network = DDQN(state_size=self.state_size,
#                             action_size=self.action_size,
#                             layer_size=hidden_size
#                             ).to(self.device)

#         self.target_net = DDQN(state_size=self.state_size,
#                             action_size=self.action_size,
#                             layer_size=hidden_size
#                             ).to(self.device)
        
        
#         self.optimizer = optim.Adam(params=self.network.parameters(), lr=1e-4)


    
    def get_action(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.network.eval()
            with torch.no_grad():
                action_values = self.network(state)
            self.network.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
        else:
            action = random.choices(np.arange(self.action_size), k=1)
        return action

    def cql_loss(self, q_values, current_action, current_batch_size):
        """Computes the CQL loss for a batch of Q-values and actions."""
        logsumexp = torch.logsumexp(q_values, dim=2, keepdim=True)

        q_a = q_values.gather(2, current_action.unsqueeze(-1).expand(current_batch_size, self.N, 1))

        return (logsumexp - q_a).sum(1).mean()

    def learn_cql(self, experiences, network, target_net, weights):
        
        states, actions, rewards, next_states, dones = experiences
        
#         print(dones)
#         pdb.set_trace()
        
        with torch.no_grad():
#             print('states',states)
#             print('actions',actions)
#             print('rewards',rewards)
#             print('next_states',next_states)
#             print('dones',dones)
            
            Q_targets_next = target_net(next_states,weights).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_a_s = network(states,weights)

        Q_expected = Q_a_s.gather(1, actions)
        
        cql1_loss = self.cql_loss(Q_a_s, actions)

        bellman_error = F.mse_loss(Q_expected, Q_targets)
        
        q1_loss = cql1_loss + 0.5 * bellman_error # alpha = 1
        
#         self.optimizer.zero_grad()
#         q1_loss.backward()
#         clip_grad_norm_(self.network.parameters(), 1.)
#         self.optimizer.step()

        # ------------------- update target network ------------------- #
#         self.soft_update(self.network, self.target_net)
        return q1_loss, cql1_loss, bellman_error
        
        
    def learn_cqr(self, experiences, network, target_net, weights):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
#         self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
#         print(weights)
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = target_net(next_states,weights).detach().cpu() #.max(2)[0].unsqueeze(1) #(batch_size, 1, N)
        
        
        
        current_batch_size = Q_targets_next.size(0)
        
        
        action_indx = torch.argmax(Q_targets_next.mean(dim=1), dim=1, keepdim=True)

        
        
        Q_targets_next = Q_targets_next.gather(2, action_indx.unsqueeze(-1).expand(current_batch_size, self.N, 1)).transpose(1,2)

        
        assert Q_targets_next.shape == (current_batch_size,1, self.N)
        # Compute Q targets for current states 
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next.to(self.device) * (1 - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected = network(states,weights).gather(2, actions.unsqueeze(-1).expand(current_batch_size, self.N, 1))
        
        Q_expected_c = network(states,weights)
        

        td_error = Q_targets - Q_expected
        assert td_error.shape == (current_batch_size, self.N, self.N), "wrong td error shape"
        huber_l = self.calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(self.quantile_tau -(td_error.detach() < 0).float()) * huber_l / 1.0
        

        
        loss = quantil_l.sum(dim=1).mean(dim=1) # , keepdim=True if per weights get multipl
        loss = loss.mean()
        
        cql1_loss = self.cql_loss(Q_expected_c, actions, current_batch_size)

        
        q1_loss = cql1_loss + 0.5 * loss # alpha = 1
        # Minimize the loss
#         q1_loss.backward()
        #clip_grad_norm_(self.qnetwork_local.parameters(),1)
#         self.optimizer.step()

        # ------------------- update target network ------------------- #
#         self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return q1_loss

    
    def calculate_huber_loss(self,td_errors, k=1.0):
        """
        Calculate huber loss element-wisely depending on kappa k.
        """
        loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
        assert loss.shape == (td_errors.shape[0], self.N, self.N), "huber loss has wrong shape"
        return loss
    

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
