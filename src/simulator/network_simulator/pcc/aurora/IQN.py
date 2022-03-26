import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class NoisyLinear(nn.Linear):
    # Noisy Linear Layer for independent Gaussian Noise
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # not trainable tensor for the nn.Module
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))

        # extra parameter for the bias and register buffer for the bias parameter
        self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
        self.register_buffer("epsilon_bias", torch.zeros(out_features))
    
        # reset parameter as initialization of the layer
        self.reset_parameter()
    
    def reset_parameter(self):
        """
        initialize the parameter of the layer and bias
        """
        std = math.sqrt(3/self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    
    def forward(self, input):
        # sample random noise in sigma weight buffer and bias buffer
        if self.training:
            self.epsilon_weight.normal_()
            bias = self.bias
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
            return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)
        else:
            return F.linear(input, self.weight, self.bias)

class IQN(nn.Module):
    def __init__(self, state_size, action_size, layer_size, n_step, seed, layer_type="ff"):
        super(IQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_shape = state_size
        self.action_size = action_size
        self.K = 32
        self.N = 8
        self.n_cos = 64
        self.layer_size = layer_size
        self.pis = torch.FloatTensor([np.pi*i for i in range(self.n_cos)]).view(1,1,self.n_cos) # Starting from 0 as in the paper 

        # layer = NoisyLinear
        layer = nn.Linear
        # self.head = nn.Linear(self.input_shape, layer_size) # cound be a cnn 
        self.head = nn.Sequential(
                nn.Linear(self.input_shape, layer_size),
                nn.ReLU(),
                nn.Linear(self.layer_size, layer_size),
            )
        self.cos_embedding = nn.Linear(self.n_cos, layer_size)
        
        self.ff_1 = layer(layer_size, layer_size)
        self.ff_2 = layer(layer_size, action_size)
        # self.ff_1 = nn.Linear(layer_size, layer_size)
        # self.ff_2 = nn.Linear(layer_size, action_size)
        #weight_init([self.head_1, self.ff_1])
        
    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1) #(batch_size, n_tau, 1)

        # Risk
        taus = taus * 0.5

        cos = torch.cos(taus*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos, taus
    
    def forward(self, input, num_tau=8):
        """
        Quantile Calculation depending on the number of tau
        
        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]
        
        """
        batch_size = input.shape[0]
        
        x = torch.relu(self.head(input))
        cos, taus = self.calc_cos(batch_size, num_tau) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.layer_size) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.layer_size)
        
        x = torch.relu(self.ff_1(x))
        out = self.ff_2(x)
        
        return out.view(batch_size, num_tau, self.action_size), taus
    
    def get_action(self, inputs):
        quantiles, _ = self.forward(inputs, self.K)
        actions = quantiles.mean(dim=1)
        return actions

    def save(self, PATH):
        torch.save(self.state_dict(),PATH)

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))