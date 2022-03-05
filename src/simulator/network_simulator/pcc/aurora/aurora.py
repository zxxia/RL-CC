###########################################################################################
# Implementation of Implicit Quantile Networks (IQN)
# Author for codes: sungyubkim, Chu Kun(kun_chu@outlook.com)
# Paper: https://arxiv.org/abs/1806.06923v1
# Reference: https://github.com/sungyubkim/Deep_RL_with_pytorch
###########################################################################################

import csv
import logging
import multiprocessing as mp
import os
from syslog import LOG_SYSLOG
import time
import types
from typing import List, Tuple, Union
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from mpi4py.MPI import COMM_WORLD

import gym
import numpy as np
import tensorflow as tf
import tqdm
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines import PPO1, logger
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.policies import FeedForwardPolicy

from simulator.network_simulator.pcc.aurora import aurora_environment
from simulator.network_simulator.pcc.aurora.schedulers import Scheduler, TestScheduler
from simulator.network_simulator.constants import BITS_PER_BYTE, BYTES_PER_PACKET
from simulator.trace import generate_trace, Trace, generate_traces
from simulator.network_simulator.pcc.aurora.replay_memory import ReplayBuffer, PrioritizedReplay
from simulator.network_simulator.pcc.aurora.IQN import IQN
from common.utils import set_tf_loglevel, pcc_aurora_reward
from plot_scripts.plot_packet_log import plot
from plot_scripts.plot_time_series import plot as plot_simulation_log


if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
    tf.contrib._warning = None

set_tf_loglevel(logging.FATAL)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import numpy as np
import math

import random
import os
import pickle
import time
from collections import deque
import matplotlib.pyplot as plt

'''Save&Load Settings'''
# check save/load
SAVE = True
LOAD = False
# paths for predction net, target net, result log
PRED_PATH = './model/iqn_pred_net.pkl'
TARGET_PATH = './model/iqn_target_net.pkl'

ACTION_MAP = [-1, -0.7, -0.45, -0.25, -0.1, 0, 0.1, 0.25, 0.45, 0.7, 1]

def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    #assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss

class IQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 n_step,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 N,
                 seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.seed_t = torch.manual_seed(seed)
        self.TAU = TAU
        self.N = N
        self.K = 32
        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = 0.9
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = 1
        
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.n_step = n_step
        self.last_action = None

        # IQN-Network
        self.qnetwork_local = IQN(state_size, action_size, layer_size, seed, N)
        self.qnetwork_target = IQN(state_size, action_size, layer_size, seed, N)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)
        
        # Replay memory
        self.memory = PrioritizedReplay(BUFFER_SIZE, self.BATCH_SIZE, seed=seed, gamma=self.GAMMA, n_step=n_step)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = self.t_step + 1
        if self.t_step % self.UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn_per(experiences)
                self.Q_updates += 1

    def choose_action(self, state, eps=0.):
        """Returns actions for given state as per current policy. Acting only every 4 frames!
        
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
            
        """
        # Epsilon-greedy action selection
        if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used!
            state = np.array(state)
            state = torch.from_numpy(state).float()
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.get_qvalues(state)#.mean(0)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            logger.log("MAX Action: ", action)
            return action
        else:
            action = random.choices(np.arange(self.action_size), k=1)
            return action

    def learn_per(self, experiences):
            """Update value parameters using given batch of experience tuples.
            Params
            ======
                experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
                gamma (float): discount factor
            """
            self.optimizer.zero_grad()
            
            states, actions, rewards, next_states, dones, idx, weights = experiences
            states = torch.FloatTensor(states)
            next_states = torch.FloatTensor(np.float32(next_states))
            actions = torch.LongTensor(actions).unsqueeze(1) 
            rewards = torch.FloatTensor(rewards).unsqueeze(1) 
            dones = torch.FloatTensor(dones).unsqueeze(1)
            weights = torch.FloatTensor(weights).unsqueeze(1)

            Q_targets_next, _ = self.qnetwork_target(next_states, self.N)
            Q_targets_next = Q_targets_next.detach() #(batch, num_tau, actions)
            q_t_n = Q_targets_next.mean(dim=1)
            # calculate log-pi 
            logsum = torch.logsumexp(\
                (Q_targets_next - Q_targets_next.max(2)[0].unsqueeze(-1))/self.entropy_tau, 2).unsqueeze(-1) #logsum trick
            assert logsum.shape == (self.BATCH_SIZE, self.N, 1), "log pi next has wrong shape"
            tau_log_pi_next = Q_targets_next - Q_targets_next.max(2)[0].unsqueeze(-1) - self.entropy_tau*logsum
                
            pi_target = F.softmax(q_t_n/self.entropy_tau, dim=1).unsqueeze(1)

            Q_target = (self.GAMMA**self.n_step * (pi_target * (Q_targets_next-tau_log_pi_next)*(1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
            assert Q_target.shape == (self.BATCH_SIZE, 1, self.N)

            q_k_target = self.qnetwork_target.get_qvalues(states).detach()
            v_k_target = q_k_target.max(1)[0].unsqueeze(-1) # (8,8,1)
            tau_log_pik = q_k_target - v_k_target - self.entropy_tau*torch.logsumexp(\
                                                                        (q_k_target - v_k_target)/self.entropy_tau, 1).unsqueeze(-1)

            assert tau_log_pik.shape == (self.BATCH_SIZE, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
            logger.log(tau_log_pik.shape)
            logger.log(actions.shape)
            munchausen_addon = tau_log_pik.gather(1, actions) #.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1)
            
            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha*torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
            assert munchausen_reward.shape == (self.BATCH_SIZE, 1, 1)
            # Compute Q targets for current states 
            Q_targets = munchausen_reward + Q_target
            # Get expected Q values from local model
            q_k, taus = self.qnetwork_local(states, self.N)
            Q_expected = q_k.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, self.N, 1))
            assert Q_expected.shape == (self.BATCH_SIZE, self.N, 1)

            # Quantile Huber loss
            td_error = Q_targets - Q_expected
            assert td_error.shape == (self.BATCH_SIZE, self.N, self.N), "wrong td error shape"
            huber_l = calculate_huber_loss(td_error, 1.0)
            quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
                
            loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)* weights # , keepdim=True if per weights get multipl
            loss = loss.mean()


            # Minimize the loss
            loss.backward()
            clip_grad_norm_(self.qnetwork_local.parameters(),1)
            self.optimizer.step()

            # ------------------- update target network ------------------- #
            self.soft_update(self.qnetwork_local, self.qnetwork_target)
            # update priorities
            td_error = td_error.sum(dim=1).mean(dim=1,keepdim=True) # not sure about this -> test 
            self.memory.update_priorities(idx, abs(td_error.data.cpu().numpy()))
            return loss.detach().cpu().numpy()            

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU*local_param.data + (1.0-self.TAU)*target_param.data)

    def save_model(self):
        # save prediction network and target network
        self.qnetwork_local.save(PRED_PATH)
        self.qnetwork_target.save(TARGET_PATH)

    def load_model(self):
        # load prediction network and target network
        self.qnetwork_local.load(PRED_PATH)
        self.qnetwork_target.load(TARGET_PATH)

def Validation(traces, dqn):
    totalR = 0
    numberR = 0

    for trace in traces:
        test_scheduler = TestScheduler(trace)
        env = gym.make('AuroraEnv-v0', trace_scheduler=test_scheduler)

        done = False
        s = np.array(env.reset())

        while not done:
            a = dqn.choose_action(s, 0)
            s, r, done, infos = env.step(ACTION_MAP[a[0]])

            totalR += r
            numberR += 1
        
    return totalR / numberR


class Aurora():
    cc_name = 'aurora'
    def __init__(self, seed: int, log_dir: str, timesteps_per_actorbatch: int,
                 pretrained_model_path: str = "", gamma: float = 0.99,
                 tensorboard_log=None, record_pkt_log: bool = False):
        self.record_pkt_log = record_pkt_log
        self.comm = COMM_WORLD
        self.seed = seed
        self.log_dir = log_dir
        self.pretrained_model_path = pretrained_model_path
        self.steps_trained = 0
        dummy_trace = generate_trace(
            (10, 10), (2, 2), (2, 2), (50, 50), (0, 0), (1, 1), (0, 0), (0, 0))
        # env = gym.make('AuroraEnv-v0', traces=[dummy_trace], train_flag=True)

    def train(self, config_file: str, total_timesteps: int,
              train_scheduler: Scheduler,
              tb_log_name: str = "", # training_traces: List[Trace] = [],
              validation_traces: List[Trace] = [],
              # real_trace_prob: float = 0
              ):

        env = gym.make('AuroraEnv-v0', trace_scheduler=train_scheduler)
        env.seed(self.seed)

        dqn = IQN_Agent(state_size=30,    
                        action_size=11,
                        layer_size=64,
                        n_step=1,
                        BATCH_SIZE=32, 
                        BUFFER_SIZE=int(1e5), 
                        LR=1e-5, 
                        TAU=1e-3, 
                        GAMMA=0.99,  
                        N=64,
                        seed=1)
        test_reward = -100

        validation_traces = []
        for i in range(20):
            validation_traces.append(Trace.load_from_file("./validation/" + str(i)))

        # model load with check
        if LOAD and os.path.isfile(PRED_PATH) and os.path.isfile(TARGET_PATH):
            dqn.load_model()
            logger.log('Load complete!')
        else:
            result = []
            logger.log('Initialize results!')

        logger.log('Collecting experience...')

        # check learning time
        start_time = time.time()
        number = 0

        EPSILON = 1.0
        # Total simulation step
        STEP_NUM = int(1e+5)
        # save frequency
        SAVE_FREQ = int(2e+2)

        for step in range(1, STEP_NUM+1):
            done = False
            s = np.array(env.reset())

            while not done:
                a = dqn.choose_action(s, EPSILON)

                # take action and get next state
                s_, r, done, infos = env.step(ACTION_MAP[a[0]])
                dqn.step(s, a[0], r, s_, done)

                # annealing the epsilon(exploration strategy)
                if number <= int(1e+4):
                    EPSILON -= 0.9/1e+4
                elif number <= int(2e+4):
                    EPSILON -= 0.09/1e+4
                
                number += 1
                s = s_

            if step % 50 == 0:
                # check time interval
                time_interval = round(time.time() - start_time, 2)

                # logger.log log
                logger.log('Used Step: ', number,
                    '| Used Trace: ', step,
                    '| Used Time:',time_interval)


            # logger.log log and save
            if step % SAVE_FREQ == 0:
                validation_reward = Validation(validation_traces, dqn)

                if validation_reward > test_reward:
                    test_reward = validation_reward
                    dqn.save_model()

                # logger.log log
                logger.log('Mean ep 100 return: ', validation_reward)

        logger.log("The training is done!")