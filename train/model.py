import ipdb
NUM_ATOMS = 9
NUM_ACTIONS = NUM_ATOMS * NUM_ATOMS * 3


from gcn import GCN
import math
import torch
import torch.nn as nn
import numpy as np
from dgl.nn.pytorch import SumPooling
from dgl import backend as F
device = torch.device("cuda")

class ActionPredictionModel(nn.Module):

    def __init__(self, node_dim,edge_dim, hid_dim, nmr_dim, max_atoms=NUM_ATOMS, bond_types=3, spec_len=1801, spec_compressed=100):
        super(ActionPredictionModel, self).__init__()

        self.max_atoms = max_atoms
        self.bond_types = bond_types
        self.node_dim = node_dim
        self.hid_dim = hid_dim
        self.nmr_dim = nmr_dim
        self.action_space_length = 3 * NUM_ATOMS * NUM_ATOMS
        self.spec_len = spec_len
        self.spec_comp = spec_compressed

        self.GCN = GCN(node_dim,edge_dim,hid_dim)
        self.sumPooling = SumPooling()
        self.compress_spec1 = nn.Linear(self.spec_len, self.spec_len//2)
        self.compress_spec2 = nn.Linear(self.spec_len//2, self.spec_comp)
        
        ## value calculation network
        self.fcv_1 = nn.Linear(self.node_dim+self.spec_comp, self.node_dim//2)
        self.fcv_2 = nn.Linear(self.node_dim//2 , 1)

        ##prior calculation Network
        self.action_1 = nn.Linear(self.hid_dim, 1)
        self.action_2 = nn.Linear(2*self.hid_dim + self.spec_comp, hid_dim)
        self.final = nn.Linear(self.hid_dim, bond_types)

    def forward(self, data):
        """
        data : dgl graph
        """
        graph = data[0].to(device)
        len_vec = torch.FloatTensor(data[1]).to(device)
        mask = data[2].to(device)
        indexmask = data[3].to(device)
        specs = data[4].to(device)


        node_features = self.GCN(graph)
        node_features_readout = self.sumPooling(graph, node_features.clone())
       
        specs = torch.squeeze(specs,1)
        specs = torch.relu(self.compress_spec1(specs))
        specs = torch.relu(self.compress_spec2(specs))

        ## Value prediction
        node_features_readout = torch.cat([node_features_readout, specs.clone()], axis=-1)
        node_features_readout = torch.relu(self.fcv_1(node_features_readout))
        node_features_readout = self.fcv_2(node_features_readout)

        ## Prior prediction
        batch_num_objs = graph.batch_num_nodes()
        len_matrix = torch.mm(len_vec.t(), len_vec).float().unsqueeze(-1)

        p_node_features = node_features.clone()
        node_features_copy = p_node_features.clone()
        X1 = p_node_features.unsqueeze(0)
        Y1 = node_features_copy.unsqueeze(1)
        X2 = X1.repeat(p_node_features.shape[0], 1, 1)
        Y2 = Y1.repeat(1, node_features_copy.shape[0], 1)
        all_pair_features = torch.cat([Y2, X2], -1)

        #Adding information about the spectra here

        batches = graph.batch_num_nodes()
        total_nodes = torch.sum(batches).item()
        canvas = torch.zeros((total_nodes, total_nodes, self.spec_comp)) #Canvas to place the correct the correct spectra at the correct place
        for idx, b in enumerate(batches):
            prev_idx = int(torch.sum(batches[:idx]).item()) #Calculating idx from where to start
            canvas[prev_idx : prev_idx + b, prev_idx : prev_idx + b] = specs[idx].repeat((b, b, 1))

        canvas = canvas.to(device)

        all_pair_features = torch.cat([all_pair_features, canvas], dim = -1)
        second_action_features = torch.relu(all_pair_features)
        second_action_features = torch.relu(self.action_2(second_action_features))
        second_action_features = self.final(second_action_features)

        # len_matrix = N x N (N total number of nodes in the batched graph)
        # second_action_features = N x N x k (k length vector has the information of whether there need to be a bond
        # between them or not)
        second_action_features = second_action_features * len_matrix
        second_action_features_idxs = second_action_features.clone()    
        indxs = (second_action_features_idxs.view(-1) != 0).nonzero().cpu()

        prev_indx = 0
        slice_indexes = []
        final_action_probabilities = torch.zeros(len(graph.batch_num_nodes()), self.action_space_length).to(device)
        # 3 x 243
        for idx, val in enumerate(graph.batch_num_nodes()):
            curr_iter = indxs[prev_indx:prev_indx + val * val * self.bond_types]
            curr_iter = torch.tensor(
                np.pad(np.array(curr_iter.squeeze(1)), (0, self.action_space_length - len(curr_iter)),
                       constant_values=-1)).to(device)
            final_action_probabilities[idx] += (curr_iter != -1).float() * (second_action_features.view(-1)[curr_iter])
            final_action_probabilities[idx] = torch.index_select(final_action_probabilities[idx],0,indexmask[idx].long())
            prev_indx = prev_indx + val * val * self.bond_types

        return torch.softmax(final_action_probabilities+mask,dim=1),node_features_readout
    
    def predict_pi(self,data,actionmask,indexmask,targetSpectra):
        try:
            with torch.no_grad():
                node_features = self.GCN(data)
                
                targetSpectra = torch.FloatTensor([[targetSpectra]])
                targetSpectra = torch.relu(self.compress_spec1(targetSpectra))
                targetSpectra = torch.relu(self.compress_spec2(targetSpectra))
                
                #print('Pi', targetSpectra.shape)
                action_mask = torch.FloatTensor(actionmask).clone()
                node_features_copy = node_features.clone()
                X1 = node_features.unsqueeze(0)
                Y1 = node_features_copy.unsqueeze(1)
                X2 = X1.repeat(node_features.shape[0], 1, 1)
                Y2 = Y1.repeat(1, node_features_copy.shape[0], 1)
                all_pair_features = torch.cat([Y2, X2], -1)

                n = node_features.shape[0]
                spec = targetSpectra.repeat(n, n, 1)

                all_pair_features = torch.cat([all_pair_features, spec], axis = -1)
                second_action_features = torch.relu(all_pair_features)
                second_action_features = torch.relu(self.action_2(second_action_features))
                second_action_features = self.final(second_action_features)
                second_action_features_idxs = second_action_features.clone()    
                final_action_probabilities = torch.zeros(1, self.action_space_length)
                
                indxs = (second_action_features_idxs.view(-1) != 0).nonzero()
                prev_indx = 0
                for idx, val in enumerate([len(data.ndata['x'])]):
                    curr_iter = indxs[prev_indx:prev_indx + val * val * self.bond_types]
                    curr_iter = torch.tensor(
                        np.pad(np.array(curr_iter.squeeze(1)), (0, self.action_space_length - len(curr_iter)),
                               constant_values=-1))
                    final_action_probabilities[idx] += (curr_iter != -1).float() * (second_action_features.view(-1)[curr_iter])
                    prev_indx = prev_indx + val * val * self.bond_types

                final_action_probabilities = final_action_probabilities.squeeze(0)
                final_action_probabilities = torch.index_select(final_action_probabilities,0,torch.Tensor(indexmask).long())
                final_action_probabilities = final_action_probabilities*action_mask
                return final_action_probabilities
        except Exception as e:
            print('Predict Pi error', e)
            pass
    def predict_V(self, data, targetSpectra ):
        try:
            with torch.no_grad():
                node_features = self.GCN(data)
                node_features_readout = self.sumPooling(data, node_features.clone())
                # concatenating feature of the entire mol with targetSpectra

                targetSpectra = torch.FloatTensor([targetSpectra])
                targetSpectra = torch.relu(self.compress_spec1(targetSpectra))
                targetSpectra = torch.relu(self.compress_spec2(targetSpectra))

                #print('Value', targetSpectra.shape)
                node_features_readout = torch.cat([node_features_readout, targetSpectra], axis=-1)
                node_features_readout = torch.relu(self.fcv_1(node_features_readout))
                node_features_readout = self.fcv_2(node_features_readout)
                return node_features_readout.flatten()
        except Exception as e:
            print('Error in predict V', e)

