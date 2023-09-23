# TODO: Determine reward threshold

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
import time


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore")


REWARD_THRESHOLD = 0.99

class Env:
    def __init__(self, molForm, targetSpectra):
        self.molForm = molForm # CNOF Molecular Formula
        self.targetSpectra = targetSpectra
        self.state = MolState(molForm) # Molecular State that has all the loose atoms
        self.forward_model = None

    def __str__(self):
        return "Current Env State: " + str(self.state) + " MolForm: " + str(self.molForm)

    def convertStrToList(self, string):
        return [int(i) for i in string.strip('][').split(', ')]

    def reset(self, ir_train_dat, idx=74753):
        try:
            self.targetmol = ir_train_dat[idx][0]  
            self.targetSpectra = ir_train_dat[idx][1]
            self.molForm = ir_train_dat[idx][2]
            print('Trying to find', ir_train_dat[idx][0])
        except Exception as e:
            print('Reset error', e)
            time.sleep(20)
        self.state = MolState(self.molForm)

    def reward(self, episode_actor, state = None, target = None):
        # Need to change for the testing code
        # Would be same as terminal_reward for our usecase in IR

        try:    
            if state is None:
                state = self.state
            if target is None:
                target = self.targetSpectra
            mol = self.state.rdmol

            pred = episode_actor.get_IR_spectrum(Chem.MolToSmiles(mol))
            target = self.targetSpectra

            pred = torch.FloatTensor(pred)
            target = torch.FloatTensor(target)
            pred /= torch.sum(pred)
            target /= torch.sum(target)

            #tmse_loss = (torch.nansum(torch.div(torch.square(pred-target), target))/len(pred)).item()
            sid = self.SIDLoss(pred, target)
            sis = 1.0/(1+sid)

            # print('Reward:', tmse_loss, 'for', Chem.MolToSmiles(mol))
            
            return sis

        except Exception as e:
            print('Reward error', e)
            print('-'*100)
            return 0 

    def terminal_reward(self, episode_actor, state = None, target = None):
        try:    
            if state is None:
                state = self.state
            if target is None:
                target = self.targetSpectra
            mol = self.state.rdmol

            pred = episode_actor.get_IR_spectrum(Chem.MolToSmiles(mol))
            target = self.targetSpectra

            pred = torch.FloatTensor(pred)
            target = torch.FloatTensor(target)
            pred /= torch.sum(pred)
            target /= torch.sum(target)

            #tmse_loss = (torch.nansum(torch.div(torch.square(pred-target), target))/len(pred)).item()
            sid = self.SIDLoss(pred, target)
            sis = 1.0/(1+sid)

            # print('Terminal loss:', tmse_loss, 'for', Chem.MolToSmiles(mol))
            
            return sis

        except Exception as e:
            print('Terminal reward error', e)
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

        if self.terminal_reward(episode_actor) > REWARD_THRESHOLD:
            #print('It is terminal due to threshold at', Chem.MolToSmiles(self.state.rdmol))
            #print('-'*100)
            return True
        else: 
            return False

            
    
    def step(self,  actionInt: int, episode_actor, state: MolState = None):
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
        reward = self.reward(episode_actor, state)
        if terminal:
            return state, reward,terminal
        else:
            return state, 0, terminal
        


