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
from stable_baselines.common.schedules import LinearSchedule

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
LEARN_FREQ = 1
# quantile numbers for IQN
N_QUANT = 8
N_ACTION = 33
# quantiles
QUANTS = np.linspace(0.0, 1.0, N_QUANT + 1)[1:]

'''Environment Settings'''
# gamma for MDP
GAMMA = 0.99


'''Training settings'''
# mini-batch size
BATCH_SIZE = 32
# learning rage
LR = 2e-6


'''Save&Load Settings'''
# check save/load
SAVE = True
LOAD = False
# paths for predction net, target net, result log
PRED_PATH = './model/iqn_pred_net_risk.pkl'
TARGET_PATH = './model/iqn_target_net_risk.pkl'


ACTION_MAP = [-1.01, -1, -0.99,
            -0.71, -0.7, -0.69,
            -0.46, -0.45, -0.44,
            -0.26, -0.25, -0.24,
            -0.11, -0.1, -0.09,
            -0.01, 0, +0.01,
            +0.09, +0.1, +0.11,
            +0.24, +0.25, +0.26,
            +0.44, +0.45, +0.46,
            +0.69, +0.7, +0.71,
            +0.99, +1, +1.01]

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        self.reset()
        self.sample()

    def reset(self):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.out_features))

    def f(self, x):
        return x.normal_().sign().mul(x.abs().sqrt())

    def sample(self):
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

    def forward(self, x):
        if self.training:
            weight = self.mu_W + self.sigma_W * self.eps_q.ger(self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)

class ConvNet(nn.Module):
    def __init__(self, alpha = 0.05):
        super(ConvNet, self).__init__()

        # Noisy
        linear = NoisyLinear

        self.phi = linear(N_QUANT, 30)
        self.fc = linear(30, 64)
        self.fc_m = linear(64, 64)
        
        # action value distribution
        self.fc_q = linear(64, N_ACTION)
        self.alpha = alpha
            
    def forward(self, x):
        batch_size = x.shape[0]

        # Rand Initlialization
        taus = torch.rand(batch_size, N_QUANT)

        # Risk
        taus = taus * self.alpha
        
        i_pi = np.pi * torch.arange(start=1, end=N_QUANT+1).view(1, 1, N_QUANT)

        # Calculate cos(i * \pi * \tau).
        cosines = torch.cos(
            taus.view(batch_size, N_QUANT, 1) * i_pi
            ).view(batch_size *  N_QUANT, N_QUANT)

        # Calculate embeddings of taus.
        # phi_j(tau) = RELU(sum(cos(π*i*τ)*w_ij + b_j))
        rand_feat = F.relu(self.phi(cosines).view(batch_size, N_QUANT, 30))

        #logger.log(rand_feat.shape)
        x = x.view(x.size(0), -1).unsqueeze(1)  # (m, 1, 30)
        #logger.log(x)
        # Zτ(x,a) ≈ f(ψ(x) @ φ(τ))a  @表示按元素相乘
        x = x * rand_feat                       # (m, N_QUANT, 30)
        #logger.log(x.shape)
        x = F.relu(self.fc_m(F.relu(self.fc(x))))           # (m, N_QUANT, 64)
        #logger.log(x.shape)

        # note that output of IQN is quantile values of value distribution
        action_value = self.fc_q(x).transpose(1, 2) # (m, N_ACTIONS, N_QUANT)

        return action_value, taus

    def set_train(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.training = True
    
    def set_test(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.training = False

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.sample()

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

        self.optimizer = torch.optim.Adam(self.pred_net.parameters(), lr=LR)

        #self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
        # replay
        self.replay_buffer = PrioritizedReplayBuffer(MEMORY_CAPACITY, 0.6)
        self.beta_schedule = LinearSchedule(2e+4, initial_p=0.4, final_p=1.0)
        
    # Update target network
    def update_target(self, target, pred, update_rate):
        # update target network parameters using predcition network
        for target_param, pred_param in zip(target.parameters(), pred.parameters()):
            target_param.data.copy_((1.0 - update_rate) \
                                    * target_param.data + update_rate*pred_param.data)
    
    def set_train(self):
        self.pred_net.set_train()
        self.target_net.set_train()
    
    def set_test(self):
        self.pred_net.set_test()
        self.target_net.set_test()

    def save_model(self):
        # save prediction network and target network
        self.pred_net.save(PRED_PATH)
        self.target_net.save(TARGET_PATH)

    def load_model(self):
        # load prediction network and target network
        self.pred_net.load('./model/iqn_pred_net.pkl')
        self.target_net.load('./model/iqn_target_net.pkl')
    
    def load_model_risk(self):
        self.pred_net.load('./model/iqn_pred_net_risk.pkl')
        self.target_net.load('./model/iqn_target_net_risk.pkl')

    def choose_action(self, x, EPSILON):
    	# x:state
        x = torch.FloatTensor(x)
        x = torch.reshape(x, (1, 30))

        # epsilon-greedy
        if np.random.uniform() >= EPSILON:
            # greedy case
            #logger.log(x)
            action_value, tau = self.pred_net(x) 	# (N_ENVS, N_ACTIONS, N_QUANT)

            #logger.log("Value: ", action_value)
            # logger.log("Tau: ", tau)

            # Min
            action_value = action_value.mean(dim=2)
            #action_value, _ = torch.min(action_value, dim=2)
            action = torch.argmax(action_value, dim=1).data.cpu().numpy()
        else:
            # random exploration case
            action = np.random.randint(0, N_ACTION)
        return int(action)

    def store_transition(self, s, a, r, s_, done):
        self.memory_counter += 1
        self.replay_buffer.add(s, a, r, s_, float(done))

    def learn(self):
        self.learn_step_counter += 1

        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.update_target(self.target_net, self.pred_net, 1e-3)
    
        # Noisy
        self.pred_net.sample_noise()
        self.target_net.sample_noise()

        # b_s, b_a, b_r, b_s_, b_d = self.replay_buffer.sample(BATCH_SIZE)     
        # b_w, b_idxes = np.ones_like(b_r), None
        # replay
        experience = self.replay_buffer.sample(BATCH_SIZE, beta=self.beta_schedule.value(self.learn_step_counter))
        (b_s, b_a, b_r, b_s_, b_d, b_w, b_idxes) = experience
            
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

        # Min
        best_actions = q_next.mean(dim=2).argmax(dim=1) 		# (m)
        #action_value, _ = torch.min(q_next, dim=2)
        #best_actions = action_value.argmax(dim = 1)

        q_next = torch.stack([q_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
        # q_nest: (m, N_QUANT)
        # q_target = R + gamma * (1 - terminate) * q_next
        q_target = b_r.unsqueeze(1) + GAMMA * (1. -b_d.unsqueeze(1)) * q_next 
        # q_target: (m, N_QUANT)
        # detach表示该Variable不更新参数
        q_target = q_target.unsqueeze(1).detach() # (m , 1, N_QUANT)

        # quantile Huber loss
        u = q_target.detach() - q_eval 		# (m, N_QUANT, N_QUANT)
        #tau = q_eval_tau.unsqueeze(0) 		# (1, N_QUANT, 1)
        # note that tau is for present quantile
        # w = |tau - delta(u<0)|

        weight = torch.abs(q_eval_tau[..., None] - u.le(0.).float()) # (m, N_QUANT, N_QUANT)
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

        # replay
        u = u.sum(dim=1).mean(dim=1,keepdim=True)
        self.replay_buffer.update_priorities(b_idxes, abs(u.data.cpu().numpy()) + 1e-6)

        return loss

def Test(config_file):
    traces = generate_traces(config_file, 20, duration=30)
    traces = generate_traces(config_file, 100, duration=30)

    iqn = DQN()
    #iqn.load_model()
    #iqn.set_test()

    iqn_risk = DQN()
    iqn_risk.load_model_risk()
    iqn_risk.set_test()
   
    rewards = [[],[]]
    dqns = [iqn, iqn_risk]

    for i in range(1, 2):
        for trace in traces:
            test_scheduler = TestScheduler(trace)
            env = gym.make('AuroraEnv-v0', trace_scheduler=test_scheduler)

            done = False
            s = np.array(env.reset())

            while not done:
                a = dqns[i].choose_action(s, 0)
                s, r, done, infos = env.step(ACTION_MAP[int(a)])

                rewards[i].append(r)

        rewards[i].sort()
        
    for ratio in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        logger.log("Ratio: ", ratio)
        # logger.log("IQN: ", rewards[0][int(ratio * len(rewards[0]))])
        logger.log("IQN: ", rewards[1][int(ratio * len(rewards[1]))])


def Validation(traces, dqn: DQN):
    dqn.set_test()
    rewards = []

    for trace in traces:
        test_scheduler = TestScheduler(trace)
        env = gym.make('AuroraEnv-v0', trace_scheduler=test_scheduler)

        done = False
        s = np.array(env.reset())

        while not done:
            a = dqn.choose_action(s, 0)
            s, r, done, infos = env.step(ACTION_MAP[int(a)])

            rewards.append(r)
        
    return sum(rewards) / len(rewards)


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
        dqn.set_train()

        test_reward = -250

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
        loss = []

        EPSILON = 1.0
        # Total simulation step
        STEP_NUM = int(1e+5)
        # save frequency
        SAVE_FREQ = int(2e+1)

        for step in range(1, STEP_NUM+1):
            done = False
            s = np.array(env.reset())

            while not done:
                # Noisy
                a = dqn.choose_action(s, 0)

                # take action and get next state
                s_, r, done, infos = env.step(ACTION_MAP[int(a)])
                s_ = np.array(s_)

                # clip rewards for numerical stability
                # clip_r = np.sign(r)

                # annealing the epsilon(exploration strategy)
                if number <= int(1e+4):
                    EPSILON -= 0.9/1e+4
                elif number <= int(2e+4):
                    EPSILON -= 0.09/1e+4
                
                number += 1

                # store the transition
                dqn.store_transition(s, a, r, s_, done)

                # if memory fill 50K and mod 4 = 0(for speed issue), learn pred net
                if (LEARN_START <= dqn.memory_counter) and (dqn.memory_counter % LEARN_FREQ == 0):
                    loss.append(dqn.learn().item())
                
                s = s_

            # logger.log log and save
            if step % SAVE_FREQ == 0:
                time_interval = round(time.time() - start_time, 2)

                # logger.log log
                logger.log('Used Step: ', dqn.memory_counter,
                    '| Used Trace: ', step,
                    '| Used Time:', time_interval,
                    '| Loss:', round(sum(loss) / len(loss), 3))

                loss = []
                validation_reward = Validation(validation_traces, dqn)

                if step > 900 and validation_reward > test_reward:
                    test_reward = validation_reward
                    dqn.save_model()
                    logger.log("Save model")

                # logger.log log
                logger.log('Mean ep 100 return: ', validation_reward)
                dqn.set_train()

        logger.log("The training is done!")