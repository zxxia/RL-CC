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
from common.utils import set_tf_loglevel, pcc_aurora_reward
from plot_scripts.plot_packet_log import plot
from plot_scripts.plot_time_series import plot as plot_simulation_log

from simulator.network_simulator.pcc.aurora.replay_memory_back import ReplayBuffer
from simulator.network_simulator.pcc.aurora.IQN import IQN
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim

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
N_QUANT = 64
N_ACTION = 4
# quantiles
QUANTS = np.linspace(0.0, 1.0, N_QUANT + 1)[1:]

'''Environment Settings'''
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
# paths for predction net, target net, result log
PRED_PATH = './model/iqn_pred_net_risk.pkl'
TARGET_PATH = './model/iqn_target_net_risk.pkl'

ACTION_MAP = [-1, -0.01, 0.01, 1]
# ACTION_MAP = [-0.8727, -0.3685, -0.1698, -0.0816, -0.04, -0.02, -0.01,   
#            0.01, 0.02, 0.04, 0.0816, 0.1698, 0.3685, 0.8727]


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 8, 8), "huber loss has wrong shape"
    return loss
    

class DQN():
    """Interacts with and learns from the environment."""

    def __init__(self):
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
            UPDATE_EVERY (int): update frequency
            seed (int): random seed
        """
        self.state_size = 30
        self.action_size = N_ACTION
        self.seed = random.seed(5)
        self.TAU = 1e-2
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = 1
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.n_step = 1

        # IQN-Network
        self.qnetwork_local = IQN(self.state_size, self.action_size, 256, self.n_step, 5)
        self.qnetwork_target = IQN(self.state_size, self.action_size, 256, self.n_step, 5)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        print(self.qnetwork_local)
        
        # Replay memory
        self.memory = ReplayBuffer(MEMORY_CAPACITY, BATCH_SIZE, 5, self.GAMMA, self.n_step)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def store_transition(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        loss = None
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = self.t_step + 1
        if self.t_step % self.UPDATE_EVERY == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                self.Q_updates += 1
        
        return loss

    def choose_action(self, state, eps=0.):
        """Returns actions for given state as per current policy. Acting only every 4 frames!
        
        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state
            
        """

        state = np.array(state)

        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local.get_action(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used!
            action = np.argmax(action_values.cpu().data.numpy())
            return action
        else:
            action = random.choice(np.arange(self.action_size))
            return action


    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # logger.log(experiences)
        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        Q_targets_next, _ = self.qnetwork_target(next_states)
        Q_targets_next = Q_targets_next.detach().max(2)[0].unsqueeze(1) # (batch_size, 1, N)
        
        # Compute Q targets for current states 
        Q_targets = rewards.unsqueeze(-1) + (self.GAMMA**self.n_step * Q_targets_next * (1. - dones.unsqueeze(-1)))
        # Get expected Q values from local model
        Q_expected, taus = self.qnetwork_local(states)
        Q_expected = Q_expected.gather(2, actions.unsqueeze(-1).expand(self.BATCH_SIZE, 8, 1))

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        assert td_error.shape == (self.BATCH_SIZE, 8, 8), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus -(td_error.detach() < 0).float()) * huber_l / 1.0
        
        loss = quantil_l.sum(dim=1).mean(dim=1) # , keepdim=True if per weights get multipl
        loss = loss.mean()

        # Minimize the loss
        loss.backward()
        #clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
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
        self.qnetwork_local.load('./model/iqn_pred_net.pkl')
        self.qnetwork_target.load('./model/iqn_target_net.pkl')
    
    def load_model_risk(self):
        self.qnetwork_local.load('./model/iqn_pred_net_risk.pkl')
        self.qnetwork_target.load('./model/iqn_target_net_risk.pkl')


def Test(config_file):
    traces = generate_traces(config_file, 20, duration=30)
    traces = generate_traces(config_file, 1, duration=30)

    distri = [0 for i in range(N_ACTION)]

    iqn = DQN()
    #iqn.load_model()
    #iqn.set_test()

    iqn_risk = DQN()
    iqn_risk.load_model_risk()
   
    rewards = [[],[]]
    dqns = [iqn, iqn_risk]


    for i in range(1, 2):
        for trace in traces:
            # logger.log(trace.bandwidths)
            test_scheduler = TestScheduler(trace)
            env = gym.make('AuroraEnv-v0', trace_scheduler=test_scheduler)

            done = False
            s = np.array(env.reset())

            while not done:
                a = dqns[i].choose_action(s, 0)
                s, r, done, infos = env.step(ACTION_MAP[int(a)])
                distri[a] += 1

                rewards[i].append(r)

                
                sender_mi = env.senders[0].history.back() #get_run_data()
                throughput = sender_mi.get("recv rate")  # bits/sec
                send_rate = sender_mi.get("send rate")  # bits/sec
                latency = sender_mi.get("avg latency")
                loss = sender_mi.get("loss ratio")
                send_ratio = sender_mi.get('send ratio')

                
                logger.log("Thp: ", throughput,
                    " | Send Rate: ", send_rate,
                    " | Action: ", ACTION_MAP[int(a)],
                    " | Send Raio: ", send_ratio,
                    " | Latency: ", latency,
                    " | Loss: ", loss,
                    " | Real Reward: ", r)
                

        rewards[i].sort()
        
    for ratio in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        logger.log("Ratio: ", ratio)
        # logger.log("IQN: ", rewards[0][int(ratio * len(rewards[0]))])
        logger.log("IQN: ", rewards[1][int(ratio * len(rewards[1]))])
    
    for i in range(N_ACTION):
        logger.log("Action ", i, " : ", distri[i])


def Validation(traces = None, config_file = None, iqn = None):

    if traces is None:
        trace = generate_traces(config_file, 1, duration=30)
    else:
        trace = traces[0]

    if iqn is None:
        iqn = DQN()
        iqn.load_model_risk()

    save_dir = './log/'
    plot_flag = True
    cc_name = 'aurora'

    RList = []

    reward_list = []
    loss_list = []
    tput_list = []
    delay_list = []
    send_rate_list = []
    ts_list = []
    action_list = []
    mi_list = []
    obs_list = []

    os.makedirs(save_dir, exist_ok=True)
    f_sim_log = open(os.path.join(save_dir, 'aurora_simulation_log.csv'), 'w', 1)
    writer = csv.writer(f_sim_log, lineterminator='\n')
    writer.writerow(['timestamp', "target_send_rate", "send_rate",
                             'recv_rate', 'latency',
                             'loss', 'reward', "action", "bytes_sent",
                             "bytes_acked", "bytes_lost", "MI",
                             "send_start_time",
                             "send_end_time", 'recv_start_time',
                             'recv_end_time', 'latency_increase',
                             "packet_size", 'min_lat', 'sent_latency_inflation',
                             'latency_ratio', 'send_ratio',
                             'bandwidth', "queue_delay",
                             'packet_in_queue', 'queue_size', "recv_ratio", "srtt"])

    test_scheduler = TestScheduler(trace)
    env = gym.make('AuroraEnv-v0', trace_scheduler=test_scheduler)
    env.seed(13)
    obs = np.array(env.reset())

    while True:
        if env.net.senders[0].got_data:
            action = iqn.choose_action(obs, 0)
        else:
            action = np.array([0])

        # get the new MI and stats collected in the MI
        # sender_mi = env.senders[0].get_run_data()
        sender_mi = env.senders[0].history.back() #get_run_data()
        throughput = sender_mi.get("recv rate")  # bits/sec
        send_rate = sender_mi.get("send rate")  # bits/sec
        latency = sender_mi.get("avg latency")
        loss = sender_mi.get("loss ratio")
        avg_queue_delay = sender_mi.get('avg queue delay')
        sent_latency_inflation = sender_mi.get('sent latency inflation')
        latency_ratio = sender_mi.get('latency ratio')
        send_ratio = sender_mi.get('send ratio')
        recv_ratio = sender_mi.get('recv ratio')
        reward = pcc_aurora_reward(
            throughput / BITS_PER_BYTE / BYTES_PER_PACKET, latency, loss,
            trace.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET,
            trace.avg_delay * 2 / 1e3)
 
        writer.writerow([
            round(env.net.get_cur_time(), 6), round(env.senders[0].pacing_rate * BITS_PER_BYTE, 0),
            round(send_rate, 0), round(throughput, 0), round(latency, 6), loss,
            round(reward, 4), action.item(), sender_mi.bytes_sent, sender_mi.bytes_acked,
            sender_mi.bytes_lost, round(sender_mi.send_end, 6) - round(sender_mi.send_start, 6),
            round(sender_mi.send_start, 6), round(sender_mi.send_end, 6),
            round(sender_mi.recv_start, 6), round(sender_mi.recv_end, 6),
            sender_mi.get('latency increase'), sender_mi.packet_size,
            sender_mi.get('conn min latency'), sent_latency_inflation,
            latency_ratio, send_ratio,
                env.links[0].get_bandwidth(
                    env.net.get_cur_time()) * BYTES_PER_PACKET * BITS_PER_BYTE,
                    avg_queue_delay, env.links[0].pkt_in_queue, env.links[0].queue_size,
                    recv_ratio, env.senders[0].srtt])

        reward_list.append(reward)
        loss_list.append(loss)
        delay_list.append(latency * 1000)
        tput_list.append(throughput / 1e6)
        send_rate_list.append(send_rate / 1e6)
        ts_list.append(env.net.get_cur_time())
        action_list.append(action.item())
        mi_list.append(sender_mi.send_end - sender_mi.send_start)
        obs_list.append(obs.tolist())            
        obs, rewards, dones, info = env.step(ACTION_MAP[int(action)])

        RList.append(rewards)

        if dones:
            break

    if f_sim_log:
        f_sim_log.close()

    with open(os.path.join(save_dir, "aurora_packet_log.csv"), 'w', 1) as f:
        pkt_logger = csv.writer(f, lineterminator='\n')
        pkt_logger.writerow(['timestamp', 'packet_event_id', 'event_type',
                                     'bytes', 'cur_latency', 'queue_delay',
                                     'packet_in_queue', 'sending_rate', 'bandwidth'])
        pkt_logger.writerows(env.net.pkt_log)

    avg_sending_rate = env.senders[0].avg_sending_rate
    tput = env.senders[0].avg_throughput
    avg_lat = env.senders[0].avg_latency
    loss = env.senders[0].pkt_loss_rate
    pkt_level_reward = pcc_aurora_reward(tput, avg_lat,loss,
            avg_bw=trace.avg_bw * 1e6 / BITS_PER_BYTE / BYTES_PER_PACKET)
    pkt_level_original_reward = pcc_aurora_reward(tput, avg_lat, loss)

    plot_simulation_log(trace, os.path.join(save_dir, 'aurora_simulation_log.csv'), save_dir, cc_name)
    bin_tput_ts, bin_tput = env.senders[0].bin_tput
    bin_sending_rate_ts, bin_sending_rate = env.senders[0].bin_sending_rate
    lat_ts, lat = env.senders[0].latencies
    plot(trace, bin_tput_ts, bin_tput, bin_sending_rate_ts,
                 bin_sending_rate, tput * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6,
                 avg_sending_rate * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6,
                 lat_ts, lat, avg_lat * 1000, loss, pkt_level_original_reward,
                 pkt_level_reward, save_dir, cc_name)

    with open(os.path.join(save_dir, "{}_summary.csv".format(cc_name)), 'w', 1) as f:
        summary_writer = csv.writer(f, lineterminator='\n')
        summary_writer.writerow([
                    'trace_average_bandwidth', 'trace_average_latency',
                    'average_sending_rate', 'average_throughput',
                    'average_latency', 'loss_rate', 'mi_level_reward',
                    'pkt_level_reward'])
        summary_writer.writerow(
                    [trace.avg_bw, trace.avg_delay,
                     avg_sending_rate * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6,
                     tput * BYTES_PER_PACKET * BITS_PER_BYTE / 1e6, avg_lat,
                     loss, np.mean(reward_list), pkt_level_reward])

    return sum(RList) / len(RList)


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

        validation_traces = []
        for i in range(1):
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
        RList = []
        AList = [0 for i in range(N_ACTION)]

        EPSILON = 1.0
        # Total simulation step
        STEP_NUM = int(1e+5)
        # save frequency
        SAVE_FREQ = int(1e+1)

        for step in range(1, STEP_NUM+1):
            done = False
            s = np.array(env.reset())

            while not done:
                # Noisy
                a = dqn.choose_action(s, EPSILON)

                # take action and get next state
                s_, r, done, infos = env.step(ACTION_MAP[int(a)])
                s_ = np.array(s_)
                RList.append(r)
                logger.log(r)
                
                AList[int(a)] += 1

                # clip rewards for numerical stability
                # clip_r = np.sign(r)

                # annealing the epsilon(exploration strategy)
                if number <= int(1e+4):
                    EPSILON -= 0.9/1e+4
                elif number <= int(2e+4):
                    EPSILON -= 0.09/1e+4
                
                number += 1

                # store the transition
                temp = dqn.store_transition(s, a, r, s_, done)

                if temp is not None:
                    loss.append(temp.item())
                
                s = s_

            # logger.log log and save
            if step % SAVE_FREQ == 0:
                time_interval = round(time.time() - start_time, 2)

                # logger.log log
                logger.log('Used Step: ', dqn.t_step,
                    '| EPSILON: ', EPSILON,
                    '| Used Trace: ', step,
                    '| Used Time:', time_interval,
                    '| Reward:', round(sum(RList) / len(RList), 3),
                    '| Loss:', round(sum(loss) / len(loss), 3))
                
                for i in range(N_ACTION):
                    logger.log(ACTION_MAP[i], ": ", AList[i])

                AList = [0 for i in range(N_ACTION)]
                loss = []
                RList = []
                validation_reward = Validation(traces = validation_traces, iqn = dqn)
                logger.log('Mean ep 100 return: ', validation_reward)
                dqn.save_model()



        logger.log("The training is done!")