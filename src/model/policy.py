import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.backends.cudnn.enabled = False
import pandas as pd


class PolicyCNN(nn.Module):
    def __init__(self, action_num, policy_mask, action_state, path_feature, link_feature, input_dim, pad_idx=None, speed_data=None):
        super(PolicyCNN, self).__init__()

        # Load speed data
        self.speed_data = speed_data

        self.policy_mask = torch.from_numpy(policy_mask).long()
        policy_mask_pad = np.concatenate([policy_mask, np.zeros((policy_mask.shape[0], 1), dtype=np.int32)], 1)
        self.policy_mask_pad = torch.from_numpy(policy_mask_pad).long()
        action_state_pad = np.concatenate([action_state, np.expand_dims(np.arange(action_state.shape[0]), 1)], 1)
        self.action_state_pad = torch.from_numpy(action_state_pad).long()
        
        self.path_feature = torch.from_numpy(path_feature).float()
        self.link_feature = torch.from_numpy(link_feature).float()
        self.new_index = torch.tensor([7, 0, 1, 6, 8, 2, 5, 4, 3]).long()
      
        self.pad_idx = pad_idx
        self.action_num = action_num

        self.conv1 = nn.Conv2d(input_dim, 20, 3, padding=1)  # [batch, 20, 3, 3]
        self.pool = nn.MaxPool2d(2, 1)  # [batch, 20, 3, 3]
        self.conv2 = nn.Conv2d(20, 30, 2)  # [batch, 30, 1, 1]
        self.fc1 = nn.Linear(30, 120)  # [batch, 120]
        self.fc2 = nn.Linear(120, 84)  # [batch, 84]
        self.fc3 = nn.Linear(84, action_num)  # [batch, 8]

        # Increase the input dimension by 1 to account for the weather feature
        self.conv1 = nn.Conv2d(input_dim + 1, 20, 3, padding=1)

    def to_device(self, device):
        self.policy_mask = self.policy_mask.to(device)
        self.policy_mask_pad = self.policy_mask_pad.to(device)
        self.action_state_pad = self.action_state_pad.to(device)
        self.path_feature = self.path_feature.to(device)
        self.link_feature = self.link_feature.to(device)
        self.new_index = self.new_index.to(device)

    def process_features(self, state, des, time_step):

        # self.action_state_pad = self.action_state_pad.to(state.device)
        state_neighbor = self.action_state_pad[state]
        # print('state_neighbor', state_neighbor)
        neigh_path_feature = self.path_feature[state_neighbor, des.unsqueeze(1).repeat(1, self.action_num + 1), :]
        # print('neigh_path_feature',neigh_path_feature)
        neigh_edge_feature = self.link_feature[state_neighbor, :]
        # print('neigh_edge_feature',neigh_edge_feature)
        neigh_mask_feature = self.policy_mask_pad[state].unsqueeze(-1)  # [batch_size, 9, 1]

        # Get speed features
        speed_features = []
        for i in range(state.size(0)):
            speed_row = []
            for neighbor in state_neighbor[i]:
                speed = self.speed_data.get((neighbor.item(), time_step[i].item()), 0)  # Default to 0 if not found
                speed_row.append(speed)
            speed_features.append(speed_row)
        speed_feature = torch.tensor(speed_features, dtype=torch.float32, device=state.device).unsqueeze(-1)
        
        # print('speed_feature',speed_feature)
        # weather_feature = neigh_path_feature[:, :, 0].unsqueeze(-1).float()
        neigh_feature = torch.cat([speed_feature, neigh_path_feature, neigh_edge_feature, neigh_mask_feature], -1)

        # neigh_feature = torch.cat([neigh_path_feature, neigh_edge_feature, neigh_mask_feature], -1)
        neigh_feature = neigh_feature[:, self.new_index, :]
        # print('neigh_feature',neigh_feature)
        
        # print('block',block)
        x = neigh_feature.view(state.size(0), 3, 3, -1)
       
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x), 0.2))
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = x.view(-1, 30)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = self.fc3(x)
        return x

    def get_action_prob(self, state, des, time_step):
        x = self.process_features(state, des, time_step)
        x = self.forward(x)
        x_mask = self.policy_mask[state]  # [batch, 8]
        x = x.masked_fill((1 - x_mask).bool(), -1e32)
        return F.softmax(x, dim=1)

    def get_action_log_prob(self, state, des, time_step):
        x = self.process_features(state, des, time_step)
        x = self.forward(x)
        x_mask = self.policy_mask[state]  # [batch, 8]
        x = x.masked_fill((1 - x_mask).bool(), -1e32)
        return F.log_softmax(x, dim=1)

    def select_action(self, state, des, time_step):
        action_prob = self.get_action_prob(state, des, time_step)
        action = torch.distributions.Categorical(action_prob).sample()
        return action

    def get_kl(self, state, des, time_step):
        action_prob1 = self.get_action_prob(state, des, time_step)
        action_prob0 = action_prob1.detach()
        kl = action_prob0 * (torch.log(action_prob0) - torch.log(action_prob1))
        return kl.sum(1, keepdim=True)

    def get_log_prob(self, state, des, actions, time_step):
        action_prob = self.get_action_prob(state, des, time_step)
        # print('action_prob',action_prob)
        return torch.log(action_prob.gather(1, actions.long().unsqueeze(1)))

    def get_fim(self, state, des, time_step):
        action_prob = self.get_action_prob(state, des, time_step)
        M = action_prob.pow(-1).view(-1).detach()
        return M, action_prob, {}
