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
        except Exception as e:
            print('Reset error', e)
            time.sleep(20)
        self.state = MolState(self.molForm, self.targetNMR)
        #print('Finished env reset')

    def reward(self, episode_actor, state = None, target = None):
        # Need to change for the testing code
        # Would be same as terminal_reward for our usecase in IR
        try:
            if state is None:
                state = self.state
            if target is None:
                target = self.targetSpectra
            state_mol = Chem.MolToSmiles(self.state.rdmol)
            state_mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(state_mol, sanitize=False), kekuleSmiles=True), sanitize=False)
            target_mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(self.targetmol, sanitize=False), kekuleSmiles=True), sanitize=False)
            substructmatch = int(target_mol.HasSubstructMatch(state_mol))
            if state.numInRdmol < state.totalNumOfAtoms:
                return substructmatch
            return substructmatch
        except:
            return 0

    def terminal_reward(self, episode_actor, state = None, target = None):
        try:    
            if state is None:
                state = self.state
            if target is None:
                target = self.targetSpectra
            mol = self.state.rdmol
            Chem.SanitizeMol(mol,Chem.SanitizeFlags.SANITIZE_FINDRADICALS|Chem.SanitizeFlags.SANITIZE_KEKULIZE|Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|Chem.SanitizeFlags.SANITIZE_SYMMRINGS,catchErrors=True)

            if state.numInRdmol < state.totalNumOfAtoms:
                return 0

            self.forward_model.predict_smiles(Chem.MolToSmiles(mol))
            predicted_IR = np.abs(np.array(predicted_IR))

            reward = 1 - wasserstein_distance(predicted_IR, target)
            
            return 2*(reward-0.5)
            
        except Exception as e:
            print('Terminal reward error', e)
            return 0 

    def invalidAction(self):
        raise Exception("Invalid Action has been chosen :(")

    def isTerminal(self, episode_actor, state: MolState = None):

        if state is None:
            state = self.state

        if sum(state.valid_actions()) == 0:
            return True

        if state.numInRdmol < state.totalNumOfAtoms:
            return False

        # target_ir = self.targetSpectra

        # mol = deepcopy(self.state.rdmol)
        # Chem.SanitizeMol(mol)

        state_mol = self.state.rdmol
        target_mol = Chem.MolFromSmiles(self.targetmol,sanitize=False)
        substructmatch = int(target_mol.HasSubstructMatch(state_mol))

        # if the present state is the target molecule
        if Chem.MolToSmiles(state_mol, canonical=True) == Chem.MolToSmiles(target_mol, canonical=True):
            return True

        # if the present state is a substructure of the target
        if substructmatch > 0:
            return False
        else:
            return True

        if self.terminal_reward(episode_actor) > REWARD_THRESHOLD:
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

        state.doStep(state._actionIntToList(actionInt))
        terminal = self.isTerminal(episode_actor, state)
        _ = state.valid_actions()
        reward = self.reward(episode_actor, state)
        if terminal:
            return state, reward,terminal
        else:
            return state, reward,terminal
        


