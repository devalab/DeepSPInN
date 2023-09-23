
NUM_ATOMS = 9
NUM_ACTIONS = NUM_ATOMS * NUM_ATOMS * 3
NUM_GPU = 4
# torch imports
import torch

# local imports
from utils.helpers import  Database, store_safely
from model import ActionPredictionModel
from mcts import  MCTS
import numpy as np
import os
import time
import sys
from environment.environment import  Env
from environment.molecule_state import MolState
from environment.chemprop_IR.smiles_predict import Forward_IR
from environment.RPNMR import predictor_new as P
from rdkit import Chem
from collections import OrderedDict
# import wandb
import ipdb
import pickle
from copy import deepcopy
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from queue import Queue
import random

BATCH_SIZE = 32
import ray
ray.init(address='auto', _redis_password='5241590000000000')

PRETRAINING_SIZE = 3000

nmr_list = None

@ray.remote
def execute_episode(ep, idx, model, valuemodel,nmr_list,episode_actor,ir_train_dat,gpu_id):
    EPISODES_OUTPUT = 'test_outputs/' + 'output_' + str(idx) + '.pkl'

    print('\n\nExecuting episode', ep, 'on GPU', gpu_id)

    # episode execution goes here
    # initialize mcts
    np.random.seed()
    env = Env([2,0,0,0], np.zeros(1801), [(5,3), (5,3)]) # dummy values to initialize the env

    timepoints = []
    a_store = []

    env.reset(ir_train_dat=ir_train_dat, idx=idx) 
    logging_target = str(env.targetmol)
    logging_steps = list()
    
    start = time.time()
    s = MolState(env.molForm, env.targetNMR)
    mcts = MCTS(root_molstate=s, root=None, model=model, valuemodel=valuemodel,na = 3*NUM_ATOMS*NUM_ATOMS,gamma=1.0)  # the object responsible for MCTS searches
    R = 0 
    n_mcts = 400

    max_leaf_r_hist = [0]
    #try:
    while True:
        # MCTS step
        forw_time = time.time()
        max_leaf_r = mcts.search(n_mcts=n_mcts, c=1, env=env, episode_actor=episode_actor)  # perform a forward search

        # Check if a good leaf has been reached, increase n_mcts otherwise
        if max(max_leaf_r_hist) > max_leaf_r:
            n_mcts = 800
            # print('Updated n_mcts to', n_mcts, max_leaf_r_hist, max_leaf_r)
        max_leaf_r_hist.append(max_leaf_r)
        
        state, pi, V,fpib, raw_prior = mcts.return_results(1)  # extract the root output
        #if(state.mol_state.rdmol.GetNumAtoms() > (sum(env.molForm)//2)):
        #    n_mcts = 800

        # Make the true step
        a = np.random.choice(len(pi), p=pi)
        a_store.append(a)

        s1, r, terminal = env.step(episode_actor, int(a))
        R = r
        print(env.state, "-", env.targetmol, "Reward:", r)#,"model prior argmax:  ",np.argmax(state.priors),"  action taken:  ",a," prior argmax:  ",np.argmax(pi),"   Episode: ",ep)
        
        sys.stdout.flush()

        logging_steps.append(str(env.state))
        if terminal:
            break
        else:
            mcts.forward(a, s1)
    del mcts

    # Finished episode
    sys.stdout.flush()

    logging_self_reward = env.reward(episode_actor, None, None, logging_target)
    logging_reward = R
    logging_time = time.time() - start

    logging_set = (logging_target, logging_self_reward, logging_steps, logging_reward, logging_time)

    if not os.path.exists(EPISODES_OUTPUT):
        empty_list = list()
        with open(EPISODES_OUTPUT, 'wb') as f:
            pickle.dump(empty_list, f, pickle.HIGHEST_PROTOCOL)

    with open(EPISODES_OUTPUT, 'r+b') as f:
        current_log = pickle.load(f)
        f.seek(0)
        current_log.append(logging_set)
        pickle.dump(current_log, f, pickle.HIGHEST_PROTOCOL)

    print('Finished episode {}, T:{} P:{}, total return: {}, total time: {} sec'.format(ep, env.targetmol, env.state, np.round(R, 2),np.round((time.time() - start), 1)))

    if ray.get(episode_actor.get_size.remote()) > PRETRAINING_SIZE:
        ray.get(episode_actor.store_trainreward.remote(R))
    else:
        ray.get(episode_actor.store_reward.remote(R))
    
    return 1


@ray.remote(num_gpus=1)
class EpisodeActor(object):
    def __init__(self, idx):
        self.idx = idx
        self.experiences = []
        self.clearFlag = False
        self.max_size = 50000
        self.size = 0
        self.insert_index = 0
        self.reward = []
        self.trainreward = []
        self.in_queue = False
        self.model_dict = []
        self.lock = False
        self.ir_forward = Forward_IR("~/DeepSPInN/models/IR_Forward" , True)
        #self.nmr_forward = P.NMRPredictor("~/DeepSPInN/models/RPNMR/best_model.meta","~/DeepSPInN/models/RPNMR/best_model.00000000.state",True)
        self.cached_ir_size = 0

        self.cached_ir_forward = {}
    
    def add_experience(self, experience):
        self.store(experience)

    def get_NMR_spectrum(self,smiles):
        return self.nmr_forward.predict(smiles)
    
    def get_IR_spectrum(self, smiles):
        smiles = Chem.CanonSmiles(smiles)
        if smiles in self.cached_ir_forward:
            return self.cached_ir_forward[smiles]
        else:
            try:
                self.cached_ir_forward[smiles] = self.ir_forward.predict_smiles(smiles)
            except Exception as e:
                print('-' * 100)
                print(e)
                print('-' * 100)
                self.cached_ir_forward[smiles] = torch.zeros(1801)
            if self.cached_ir_size < 1000: # don't add spectra to cache after these many spectra
                self.cached_ir_size += 1
                # print(self.idx, "has", self.new_spectra, "new spectra")
            else:
                self.cached_ir_size = 1
                self.cached_ir_forward.clear()
                self.cached_ir_forward[smiles] = self.ir_forward.predict_smiles(smiles)

            return self.cached_ir_forward[smiles]

    def store(self,experience):
        if self.size < self.max_size:
            self.experiences.append(experience)
            self.size +=1
        else:
            self.experiences[self.insert_index] = experience
            self.insert_index += 1
            if self.insert_index >= self.size:
                self.insert_index = 0
    
    def clear_exprience(self):
        self.messages = []

    def get_experience(self):
        experiences = self.experiences
        return experiences

    def get_size(self):
        return self.size

    def increment_size(self):
        self.size += 1

    def set_lock(self,value):
        self.lock = value
        return value

    def get_lock(self):
        return self.lock

    def set_queue(self,value):
        self.in_queue = value
        return value

    def get_queue(self):
        return self.in_queue

    def add_model(self,value):
        self.model_dict.append(value)

    def get_model(self):
        return self.model_dict.pop()

    def get_model_length(self):
        return len(self.model_dict)

    def store_reward(self,value):
        self.reward.append(value)
        return 1

    def empty_reward(self):
        self.reward = []
        return 1

    def get_reward(self):
        return self.reward

    def store_trainreward(self,value):
        self.trainreward.append(value)
        return 1

    def empty_trainreward(self):
        self.trainreward = []
        return 1

    def get_trainreward(self):
        return self.trainreward

def run():
    #THES REWARD FILES NOT USED ANYMORE:
    # reward_file = open("./reward.log","w")
    # train_reward_file = open("./trainreward.log","w")
    reward_file, train_reward_file = None, None
    manager = mp.Manager()
    database = manager.dict()

    episode_actors = [EpisodeActor.remote(i) for i in range(NUM_GPU)]

    with open('~/DeepSPInN/data/qm9_clean_ir_nmr.pickle', 'rb') as handle:
        ir_all_datasets = pickle.load(handle)
    ir_train_dat = ir_all_datasets["test"]

    # remote_ap = APModel.remote(True)
    model =  ActionPredictionModel(88, 6, 88, 64)
    model.load_state_dict(torch.load("../train/saved_models/prior21.state",map_location='cpu'))
    model_episode =  ActionPredictionModel(88, 6, 88, 64)
    model_episode.load_state_dict(deepcopy(model.state_dict()))

    valuemodel =  "valuemodel"
    valuemodel_episode =  "valuemodel"
    ### --------------models------------------ ###

    num_processes = 5
    num_episodes = 2

    for mol_id in range(3000,4976):
    # tricky_mols = [1, 9, 13, 57, 59, 79, 82, 95, 108, 110, 167, 169, 174, 187, 194, 197]
    # for mol_id in tricky_mols:
        # each molecule runs for num_episodes * num_processes times
        for i in range(num_episodes):
            for gpu_id in range(NUM_GPU):
                ray.get([episode_actors[gpu_id].set_lock.remote(True)])
            results = [execute_episode.remote(j, mol_id, model_episode, valuemodel_episode,nmr_list,episode_actors[gpu_id],ir_train_dat,gpu_id) for j in range(num_processes*i,num_processes*(i+1)) for gpu_id in range(NUM_GPU)]
            ready_ids, remaining_ids = ray.wait(results,num_returns=num_processes*NUM_GPU)
            # print("-----Episode", i, "done----")
            for gpu_id in range(NUM_GPU):
                ray.get(episode_actors[gpu_id].set_lock.remote(False))

    # train_process.join()
if __name__ == '__main__':
    run()
