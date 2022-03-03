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
from simulator.network_simulator.pcc.aurora.replay_memory import ReplayBuffer, PrioritizedReplayBuffer
from common.utils import set_tf_loglevel, pcc_aurora_reward
from plot_scripts.plot_packet_log import plot
from plot_scripts.plot_time_series import plot as plot_simulation_log


if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
    tf.contrib._warning = None

set_tf_loglevel(logging.FATAL)

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import random
import os
import pickle
import time
from collections import deque
import matplotlib.pyplot as plt

# Parameters
import argparse


'''DQN settings'''
# target policy sync interval
TARGET_REPLACE_ITER = 1
# simulator steps for start learning
LEARN_START = int(1e+3)
# (prioritized) experience replay memory size
MEMORY_CAPACITY = int(1e+5)
# simulator steps for learning interval
LEARN_FREQ = 4
# quantile numbers for IQN
N_QUANT = 64
# quantiles
QUANTS = np.linspace(0.0, 1.0, N_QUANT + 1)[1:]

'''Environment Settings'''
# Total simulation step
STEP_NUM = int(1e+8)
# gamma for MDP
GAMMA = 0.99


'''Training settings'''
# mini-batch size
BATCH_SIZE = 32
# learning rage
LR = 1e-4


'''Save&Load Settings'''
# check save/load
SAVE = True
LOAD = False
# save frequency
SAVE_FREQ = int(1e+3)
# paths for predction net, target net, result log
PRED_PATH = './data/model/iqn_pred_net.pkl'
TARGET_PATH = './data/model/iqn_target_net.pkl'
RESULT_PATH = './data/plots/iqn_result.pkl'


ACTION_MAP = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]

# # define huber function
# def huber(x):
# 	cond = (c.abs()<1.0).float().detach()
# 	return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1.0 - cond)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.phi = nn.Linear(1, 30, bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(30))
        self.fc = nn.Linear(30, 64)
        
        # action value distribution
        self.fc_q = nn.Linear(64, 11) 
        
        # Initialization 
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                
            
    def forward(self, x):
        # Rand Initlialization
        tau = torch.rand(N_QUANT, 1) # (N_QUANT, 1)
        # Quants=[1,2,3,...,N_QUANT]
        quants = torch.arange(0, N_QUANT, 1.0) # (N_QUANT,1)

        # phi_j(tau) = RELU(sum(cos(π*i*τ)*w_ij + b_j))
        cos_trans = torch.cos(quants * tau * 3.141592).unsqueeze(2) # (N_QUANT, N_QUANT, 1)
        rand_feat = F.relu(self.phi(cos_trans).mean(dim=1) + self.phi_bias.unsqueeze(0)).unsqueeze(0) 
        # (1, N_QUANT, 7 * 7 * 64)
        #logger.log(rand_feat.shape)
        x = x.view(x.size(0), -1).unsqueeze(1)  # (m, 1, 7 * 7 * 64)
        #logger.log(x)
        # Zτ(x,a) ≈ f(ψ(x) @ φ(τ))a  @表示按元素相乘
        x = x * rand_feat                       # (m, N_QUANT, 7 * 7 * 64)
        #logger.log(x.shape)
        x = F.relu(self.fc(x))                  # (m, N_QUANT, 512)
        #logger.log(x.shape)

        # note that output of IQN is quantile values of value distribution
        action_value = self.fc_q(x).transpose(1, 2) # (m, N_ACTIONS, N_QUANT)

        return action_value, tau


    def save(self, PATH):
        torch.save(self.state_dict(),PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))

class DQN(object):
    def __init__(self):
        self.pred_net, self.target_net = ConvNet(), ConvNet()
        # sync evac target
        self.update_target(self.target_net, self.pred_net, 1.0)
            
        # simulator step counter
        self.memory_counter = 0
        # target network step counter
        self.learn_step_counter = 0

        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)
        
    # Update target network
    def update_target(self, target, pred, update_rate):
        # update target network parameters using predcition network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate*pred_param.data)
    
    def save_model(self):
        # save prediction network and target network
        self.pred_net.save(PRED_PATH)
        self.target_net.save(TARGET_PATH)

    def load_model(self):
        # load prediction network and target network
        self.pred_net.load(PRED_PATH)
        self.target_net.load(TARGET_PATH)

    def choose_action(self, x, EPSILON):
    	# x:state
        x = torch.FloatTensor(x)
        x = torch.reshape(x, (1, 30))

        # epsilon-greedy
        if np.random.uniform() >= EPSILON:
            # greedy case
            #logger.log(x)
            action_value, tau = self.pred_net(x) 	# (N_ENVS, N_ACTIONS, N_QUANT)
            #logger.log(action_value)
            action_value = action_value.mean(dim=2)
            #logger.log(action_value)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
            #logger.log(action)
        else:
            # random exploration case
            action = np.random.randint(0, 11)
        return int(action)

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
        self.learn_step_counter += 1
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 1e-2)
    
        b_s, b_a, b_r, b_s_, b_d = self.replay_buffer.sample(BATCH_SIZE)
        b_w, b_idxes = np.ones_like(b_r), None
            
        b_s = torch.FloatTensor(b_s)
        b_a = torch.LongTensor(b_a)
        b_r = torch.FloatTensor(b_r)
        b_s_ = torch.FloatTensor(b_s_)
        b_d = torch.FloatTensor(b_d)

        # action value distribution prediction
        q_eval, q_eval_tau = self.pred_net(b_s) 	# (m, N_ACTIONS, N_QUANT), (N_QUANT, 1)
        mb_size = q_eval.size(0)
        # squeeze去掉第一维
        # torch.stack函数是将矩阵进行叠加，默认dim=0，即将[]中的n个矩阵变成n维
        # index_select函数是进行索引查找。
        q_eval = torch.stack([q_eval[i].index_select(0, b_a[i]) for i in range(mb_size)]).squeeze(1) 
        # (m, N_QUANT)
        # 在q_eval第二维后面加一个维度
        q_eval = q_eval.unsqueeze(2) 				# (m, N_QUANT, 1)
        # note that dim 1 is for present quantile, dim 2 is for next quantile
        
        # get next state value
        q_next, q_next_tau = self.target_net(b_s_) 				# (m, N_ACTIONS, N_QUANT), (N_QUANT, 1)
        best_actions = q_next.mean(dim=2).argmax(dim=1) 		# (m)
        q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
        # q_nest: (m, N_QUANT)
        # q_target = R + gamma * (1 - terminate) * q_next
        q_target = b_r.unsqueeze(1) + GAMMA * (1. -b_d.unsqueeze(1)) * q_next 
        # q_target: (m, N_QUANT)
        # detach表示该Variable不更新参数
        q_target = q_target.unsqueeze(1).detach() # (m , 1, N_QUANT)

        # quantile Huber loss
        u = q_target.detach() - q_eval 		# (m, N_QUANT, N_QUANT)
        tau = q_eval_tau.unsqueeze(0) 		# (1, N_QUANT, 1)
        # note that tau is for present quantile
        # w = |tau - delta(u<0)|
        weight = torch.abs(tau - u.le(0.).float()) # (m, N_QUANT, N_QUANT)
        loss = F.smooth_l1_loss(q_eval, q_target.detach(), reduction='none')
        # (m, N_QUANT, N_QUANT)
        loss = torch.mean(weight * loss, dim=1).mean(dim=1)
        
        # calculate importance weighted loss
        b_w = torch.Tensor(b_w)
        loss = torch.mean(b_w * loss)
        
        # backprop loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

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
        test_scheduler = TestScheduler(dummy_trace)
        env = gym.make('AuroraEnv-v0', trace_scheduler=test_scheduler)
        self.model = DQN()

    def train(self, config_file: str, total_timesteps: int,
              train_scheduler: Scheduler,
              tb_log_name: str = "", # training_traces: List[Trace] = [],
              validation_traces: List[Trace] = [],
              # real_trace_prob: float = 0
              ):

        env = gym.make('AuroraEnv-v0', trace_scheduler=train_scheduler)
        env.seed(self.seed)

        dqn = DQN()

        # model load with check
        if LOAD and os.path.isfile(PRED_PATH) and os.path.isfile(TARGET_PATH):
            dqn.load_model()
            pkl_file = open(RESULT_PATH,'rb')
            result = pickle.load(pkl_file)
            pkl_file.close()
            logger.log('Load complete!')
        else:
            result = []
            logger.log('Initialize results!')

        logger.log('Collecting experience...')

        # episode step for accumulate reward 
        epinfobuf = deque(maxlen=100)
        # check learning time
        start_time = time.time()

        # env reset
        s = np.array(env.reset())

        EPSILON = 1.0

        for step in range(1, STEP_NUM+1):
            a = dqn.choose_action(s, EPSILON)

            # take action and get next state
            s_, r, done, infos = env.step(ACTION_MAP[int(a)])
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfobuf.append(maybeepinfo)
            s_ = np.array(s_)

            # clip rewards for numerical stability
            clip_r = np.sign(r)

            # store the transition
            dqn.store_transition(s, a, clip_r, s_, done)

            # annealing the epsilon(exploration strategy)
            if step <= int(1e+4):
                EPSILON -= 0.9/1e+4
            elif step <= int(2e+4):
                EPSILON -= 0.09/1e+4

            # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
            if (LEARN_START <= dqn.memory_counter) and (dqn.memory_counter % LEARN_FREQ == 0):
                loss = dqn.learn()

            # logger.log log and save
            if step % SAVE_FREQ == 0:
                # check time interval
                time_interval = round(time.time() - start_time, 2)
                # calc mean return
                mean_100_ep_return = round(np.mean([epinfo['r'] for epinfo in epinfobuf]),2)
                result.append(mean_100_ep_return)
                # logger.log log
                logger.log('Used Step: ',dqn.memory_counter,
                    '| EPS: ', round(EPSILON, 3),
                    '| Loss: ', loss,
                    '| Mean ep 100 return: ', r,
                    '| Used Time:',time_interval)
                # save model
                #dqn.save_model()
                #pkl_file = open(RESULT_PATH, 'wb')
                #pickle.dump(np.array(result), pkl_file)
                #pkl_file.close()

            s = s_

        logger.log("The training is done!")