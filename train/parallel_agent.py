
NUM_ATOMS = 9
NUM_ACTIONS = NUM_ATOMS * NUM_ATOMS * 3
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
from train import train
# from train import train_value
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


PRETRAINING_SIZE = 3000
# PRETRAINING_SIZE = 10

import ray
import logging
ray.init(address='auto', _redis_password='5241590000000000', logging_level=logging.DEBUG)


nmr_list = None

@ray.remote
class Agent(object):

    def __init__(self, num_episodes = 100000, n_mcts=1 ,max_ep_len=20 ,lr= 1e-4, c=4 ,gamma=1.0 ,data_size=10000,
                batch_size=32,temp=1,):
        #initialise hyper params.
        self.num_episodes = num_episodes
        self.n_mcts = n_mcts
        self.max_ep_len = max_ep_len
        self.lr = lr
        self.c = c
        self.gamma =  gamma
        self.data_size = data_size
        self.batch_size = batch_size
        self.temp = temp
        self.t_total = 0
        self.episode_returns = []
        self.R_best = -np.Inf



@ray.remote
def execute_episode(ep, idx, model, valuemodel,nmr_list,episode_actor,ir_train_dat):
    print('\n\nExecuting episode', ep)
    start = time.time()

    # episode execution goes here
    # initialize mcts
    np.random.seed()
    env = Env([2,0,0,0], np.zeros(1801), [(5,3), (5,3)]) # dummy values to initialize the env
    episode_rewards = []

    timepoints = []
    a_store = []
    exps = []
    
    idx = np.random.randint(0, len(ir_train_dat))
    #"idx = np.random.randint(0, 51)
    env.reset(ir_train_dat=ir_train_dat, idx=idx) # testing for benzene
    
    s = MolState(env.molForm, env.targetNMR)
    mcts = MCTS(root_molstate=s, root=None, model=model, valuemodel=valuemodel,na = 3*NUM_ATOMS*NUM_ATOMS,gamma=1.0)  # the object responsible for MCTS searches
    R = 0 
    n_mcts = 100
    while True:
        # MCTS step
        #debugprint('Before forward search')
        mcts.search(n_mcts=n_mcts, c=4, Env=env, episode_actor=episode_actor)  # perform a forward search

        #debugprint('Before extracting root output')
        state, pi, V,fpib = mcts.return_results(1)  # extract the root output
        if(state.mol_state.rdmol.GetNumAtoms() > (sum(env.molForm)//2)):
            n_mcts = 400

        exps.append(({'mol_graph':state.mol_graph,'action_mask':state.action_mask,'action_1_mask':state.action_1_mask,'index_mask':state.index_mask, "targetSpectra": env.targetSpectra}, V, pi,fpib))

        # Make the true step
        a = np.random.choice(len(pi), p=pi)
        a_store.append(a)
        #debugprint('Before doing step')
        s1, r, terminal = env.step(int(a), episode_actor)
        print(env.state,"  Reward:",r,"model prior argmax:  ",np.argmax(state.priors),"  action taken:  ",a," prior argmax:  ",np.argmax(pi),"   Episode: ",ep)
        sys.stdout.flush()

        R = r
        if terminal:
            break
        else:
            mcts.forward(a, s1)
    del mcts


    #debugprint('Trying to print experiences')
    for exp in exps:
        #print('Adding experience')
        # print('Adding experience', exp)
        episode_actor.add_experience.remote(tuple(exp))

    # Finished episode
    episode_rewards.append(R)  # store the total episode return
    sys.stdout.flush()


    print('Finished episode {}, total return: {}, total time: {} sec'.format(ep, np.round(R, 2),np.round((time.time() - start), 1)))
    # Train
    print("DB Size: ",ray.get(episode_actor.get_size.remote()))
    if ray.get(episode_actor.get_size.remote()) > PRETRAINING_SIZE:
        ray.get(episode_actor.store_trainreward.remote(R))
    else:
        ray.get(episode_actor.store_reward.remote(R))
    return 1




@ray.remote(num_gpus=1)
class EpisodeActor(object):
    def __init__(self):
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
        self.ir_forward = Forward_IR("../models/IR_Forward" , True)
        self.cached_ir_size = 0

        try:
            with open('../data/ir_cache_' +  str(self.idx) + '.pkl', 'rb') as handle:
                self.cached_ir_forward = pickle.load(handle)
        except Exception as e:
            self.cached_ir_forward = {}
        self.new_spectra = 0
    
    def add_experience(self, experience):
        self.store(experience)

    def get_IR_spectrum(self, smiles):
        if smiles in self.cached_ir_forward:
            return self.cached_ir_forward[smiles]

        else:
            self.cached_ir_forward[smiles] = self.ir_forward.predict_smiles(smiles)
            if self.cached_ir_size < 1000: # don't add spectra to cache after these many spectra
                self.cached_ir_size += 1
                self.new_spectra += 1
                # print(self.idx, "has", self.new_spectra, "new spectra")
                if self.new_spectra > 50:
                    # print(self.idx, "dumped")
                    with open('../data/ir_cache_' +  str(self.idx) + '.pkl', 'wb') as handle:
                        pickle.dump(self.cached_ir_forward, handle, pickle.HIGHEST_PROTOCOL)
                    self.new_spectra = 0
            else:
                self.cached_ir_size = 0
                self.new_spectra = 0
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
    num_episodes = 4000000000
    agent_obj = Agent.remote()
    num_processes = 35
    manager = mp.Manager()
    database = manager.dict()

    episode_actor = EpisodeActor.remote()

    with open('../data/qm9_train_test_val_ir_nmr.pickle', 'rb') as handle:
    # with open('../data/qm9_small_train_big_test.pickle', 'rb') as handle:
        ir_all_datasets = pickle.load(handle)
    ir_train_dat = ir_all_datasets["train"]

    ### --------------models------------------ ###
    model =  ActionPredictionModel(88, 6, 88, 64)
    #model.load_state_dict(torch.load("saved_models/priors_ir_nmr/prior27.state",map_location='cpu'))
    model_episode =  ActionPredictionModel(88, 6, 88, 64)
    model_episode.load_state_dict(deepcopy(model.state_dict()))
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    valuemodel =  "valuemodel"
    valuemodel_episode =  "valuemodel"
    ### --------------models------------------ ###

    episode_counter = 0
    [train.remote(model,valuemodel,episode_actor,"wandb",)]
    for i in range(num_episodes):
        processes = []
        ready_ids, remaining_ids = ray.wait([episode_actor.set_lock.remote(True)],num_returns=1)
        results = [execute_episode.remote(j,0,model_episode, valuemodel_episode,nmr_list,episode_actor,ir_train_dat) for j in range(num_processes*i,num_processes*(i+1))]
        ready_ids, remaining_ids = ray.wait(results,num_returns=num_processes)
        # print("-----Episode", i, "done----")
        ray.get(episode_actor.set_lock.remote(False))

        db_size = ray.get(episode_actor.get_size.remote())
        print('Main Run Loop DB Size:', db_size)
        if db_size >= PRETRAINING_SIZE:
            while not ray.get(episode_actor.get_queue.remote()):
                True
            if ray.get(episode_actor.get_queue.remote()):
                model_state_dict, valuemodel_state_dict = ray.get(episode_actor.get_model.remote())
                #time.sleep(0.2)
                print("------------model received--------------")
                print(model_state_dict['final.bias'])
                print("------------model received--------------")
                model_episode.load_state_dict(model_state_dict)
                ray.wait([episode_actor.set_queue.remote(False)],num_returns=1)
                ray.wait([episode_actor.set_lock.remote(True)],num_returns=1)
            rewards = ray.get(episode_actor.get_trainreward.remote())
            # for i in rewards:
            #    train_reward_file.write("Train Episode Reward:  " + str(i))
            print('Train Rewards:', rewards)
            ray.get(episode_actor.empty_trainreward.remote())
        else:
            rewards = ray.get(episode_actor.get_reward.remote())
            # for i in rewards:
            #     reward_file.write("Episode Reward:  " + str(i))
            #print('Episode Rewards:', rewards)
            ray.get(episode_actor.empty_reward.remote())


    # train_process.join()
if __name__ == '__main__':
    run()
