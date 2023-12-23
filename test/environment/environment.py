import sys
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
import numpy as np
import copy
import random
from collections import OrderedDict
from rdkit.Chem import rdMolDescriptors as rdDesc
from .molecular_graph import *
from .molecule_state import MolState
import torch
import warnings
from rdkit.Chem.QED import qed
from rdkit.Chem import AllChem
from rdkit import DataStructs
from scipy.stats import wasserstein_distance
import pickle
from .chemprop_IR.smiles_predict import get_IR_prediction, Forward_IR
import ipdb
import ray
import time


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")


REWARD_THRESHOLD = 0.95

class Env:
    def __init__(self, molForm, targetIRSpectra, targetNMRSpectra):
        self.molForm = molForm # CNOF Molecular Formula
        self.targetSpectra = targetIRSpectra
        self.targetNMR = targetNMRSpectra
        self.state = MolState(molForm, targetNMRSpectra) # Molecular State that has all the loose atoms

    def __str__(self):
        return "Current Env State: " + str(self.state) + " MolForm: " + str(self.molForm)

    def convertStrToList(self, string):
        return [int(i) for i in string.strip('][').split(', ')]

    def reset(self, ir_train_dat, idx=74753):
        try:
            self.targetmol = ir_train_dat[idx][0]  
            self.molForm = ir_train_dat[idx][1]
            self.targetSpectra = ir_train_dat[idx][2] # ir-spectra
            self.targetNMR = ir_train_dat[idx][3] # nmr-spectra
            print('Trying to find', self.targetmol)
        except Exception as e:
            print('Reset error', e)
            time.sleep(20)
        self.state = MolState(self.molForm, self.targetNMR, self.targetmol)

    def reward(self, episode_actor, state = None, target = None, onlySmiles = None):
        try:    
            if onlySmiles is None:
                if state is None:
                    state = self.state
                mol = self.state.rdmol
            else:
                mol = Chem.MolFromSmiles(onlySmiles)

            if target is None:
                target = self.targetSpectra

            pred = ray.get(episode_actor.get_IR_spectrum.remote(Chem.MolToSmiles(mol)))
            target = self.targetSpectra

            pred = torch.FloatTensor(pred)
            target = torch.FloatTensor(target)
            pred /= torch.sum(pred)
            target /= torch.sum(target)

            #tmse_loss = (torch.nansum(torch.div(torch.square(pred-target), target))/len(pred)).item()
            sid = self.SIDLoss(pred, target)
            sis = 1.0/(1+sid)


            # NMR Loss
#            nmr_dist = 0
#            if state.numInRdmol == state.totalNumOfAtoms and onlySmiles is not None:
#                nmr_pred = ray.get(episode_actor.get_NMR_spectrum.remote(Chem.MolToSmiles(mol)))
#                if nmr_pred != -1:
#                    nmr_pred = np.array(nmr_pred)
#                    nmr_pred /= 220
#                    
#                    nmr_target = np.array([x[0] for x in self.targetNMR])
#                    nmr_target = np.array(nmr_target)
#                    nmr_target /= 220
#
#                    nmr_pred = sorted(nmr_pred)
#                    nmr_target = sorted(nmr_target)
#                    diff_in_atoms = len(nmr_target) - len(nmr_pred)
#
#                    if diff_in_atoms != 0:  # !!!!!!!!!------!!!!!
#                        return 0
#
#                    nmr_pred = nmr_pred[:len(nmr_target)]
#                    nmr_dist = 1 - wasserstein_distance(nmr_pred, nmr_target)
#           
#            sis = (sis + nmr_dist) / 2

            return sis

            # present_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            # present_mol.UpdatePropertyCache()
            # present_sdtq = [0, 0, 0, 0]
            # for atom in present_mol.GetAtoms():
            #     if atom.GetSymbol() == 'C':
            #         h_count = atom.GetTotalNumHs()
            #         if h_count >=0 and h_count <= 3:
            #             present_sdtq[h_count] += 1


            # target_splits = [x[1] for x in self.targetNMR]
            # target_sdtq = [target_splits.count(0), target_splits.count(1), target_splits.count(2), target_splits.count(3)]
            # 
            # if target_sdtq[3] > present_sdtq[3]: # you needed to have more -CH3
            #     sis = 0
            # elif target_sdtq[3] == present_sdtq[3] and target_sdtq[2] > present_sdtq[2]: # nice -CH3, but you needed to have more -CH2
            #     sis = 0
            # elif target_sdtq[3] == present_sdtq[3] and target_sdtq[2] == present_sdtq[2] and target_sdtq[1] > present_sdtq[1]:
            #     sis = 0
           
            # if target_sdtq != present_sdtq and sis!=0:
            #     sis = 0
            #     # print('Set SIS to 0 for', Chem.MolToSmiles(mol), 'for target', self.targetmol)

            # return sis

        except Exception as e:
            print('Reward error', e)
            print('-'*100)
            return 0 

    def SIDLoss(self, a, b):
        # a is predicted, b is target
        a = torch.FloatTensor(a)
        b = torch.FloatTensor(b)
        threshold = 1e-8
        nan_mask=torch.isnan(b)+torch.isnan(a)
        zero_sub=torch.zeros_like(b)

        a[a < threshold] = threshold
        sum_model_spectra = torch.sum(torch.where(nan_mask,zero_sub,a))
        a = torch.div(a,sum_model_spectra)

        b[b < threshold] = threshold
        sum_target_spectra = torch.sum(torch.where(nan_mask,zero_sub,b))
        b = torch.div(b,sum_target_spectra)

        loss = torch.ones_like(b)
        a[nan_mask]=1
        b[nan_mask]=1
        loss = torch.mul(torch.log(torch.div(a,b)),a) + torch.mul(torch.log(torch.div(b,a)),b)
        loss[nan_mask]=0
        loss = torch.sum(loss)

        return loss.item()

    def invalidAction(self):
        raise Exception("Invalid Action has been chosen :(")

    def isTerminal(self, episode_actor, state: MolState = None):

        if state is None:
            state = self.state

        if sum(state.valid_actions()) == 0:
            return True

        if state.numInRdmol < state.totalNumOfAtoms:
            return False

        treward = ray.get(episode_actor.get_cache_reward.remote(Chem.MolToSmiles(state.rdmol)))
        if treward is None:
            treward = self.reward(episode_actor, state)
            ray.get(episode_actor.set_cache_reward.remote(Chem.MolToSmiles(state.rdmol), treward))
        if treward > REWARD_THRESHOLD:
            return True

        mol = self.state.rdmol
        
        present_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        present_mol.UpdatePropertyCache()
        present_sdtq = [0, 0, 0, 0]
        for atom in present_mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                h_count = atom.GetTotalNumHs()
                if h_count >=0 and h_count <= 3:
                    present_sdtq[h_count] += 1


        target_splits = [x[1] for x in self.targetNMR]
        target_sdtq = [target_splits.count(0), target_splits.count(1), target_splits.count(2), target_splits.count(3)]

        isTerminal = False

        if target_sdtq[3] > present_sdtq[3]: # you needed to have more -CH3
            isTerminal = True
        elif target_sdtq[3] == present_sdtq[3] and target_sdtq[2] > present_sdtq[2]: # nice -CH3, but you needed to have more -CH2
            isTerminal = True
        elif target_sdtq[3] == present_sdtq[3] and target_sdtq[2] == present_sdtq[2] and target_sdtq[1] > present_sdtq[1]:
            isTerminal = True
        elif target_sdtq[3] == present_sdtq[3] and target_sdtq[2] == present_sdtq[2] and target_sdtq[1] == present_sdtq[1] and target_sdtq[0] == present_sdtq[0]:
           isTerminal = True

        if present_sdtq[0] > target_sdtq[0]: # present state has too many -#C
            isTerminal = True
        elif present_sdtq[0] == target_sdtq[0] and present_sdtq[1] > target_sdtq[1]:
            isTerminal = True
        elif present_sdtq[0] == target_sdtq[0] and present_sdtq[1] == target_sdtq[1] and present_sdtq[2] > target_sdtq[2]:
            isTerminal = True    
    
        return isTerminal

            
    
    def step(self, episode_actor, actionInt: int, state: MolState = None):
        if state is None:
            state = self.state
        
        valid_actions = state.action_mask
        if valid_actions[actionInt] == 0:
            action = state._actionIntToList(actionInt)
            #print(state._actionIntToList(actionInt))
            #print(state)

            return self.invalidAction()

        before_step = time.time()
        state.doStep(state._actionIntToList(actionInt))
        after_step = time.time()
        #print('Time for doing step:', after_step-before_step)

        terminal = self.isTerminal(episode_actor, state)
        after_terminal = time.time()
        #print('Time for checking terminal:', after_terminal-after_step)

        _ = state.valid_actions()
        if terminal:
            target_mol = Chem.MolFromSmiles(state.targetSmiles)
            if state.numInRdmol < target_mol.GetNumAtoms():
                reward = 0
            else:
                reward = ray.get(episode_actor.get_cache_reward.remote(Chem.MolToSmiles(state.rdmol)))
                if reward is None:
                    reward = self.reward(episode_actor, state)
                    ray.get(episode_actor.set_cache_reward.remote(Chem.MolToSmiles(state.rdmol), reward))
        else:
            reward = 0
        return state, reward, terminal
        


