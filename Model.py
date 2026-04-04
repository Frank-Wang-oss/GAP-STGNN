import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from collections import OrderedDict
from Model_Base import *







#### Best for RUL
class FC_STGNN_REG(nn.Module):
    def __init__(self, lstmhidden_dim, lstmout_dim, conv_kernel, hidden_dim, num_patch_max, input_length, num_node, window_kernel,stride,decay, pooling_choice, n_class, tem_map, patch_dim):
        super(FC_STGNN_REG, self).__init__()
        self.adap_Patch = BinaryGaussianPatchLayer_v3(num_patch_max, input_length=input_length,
                                                          num_channels=num_node)
        self.num_patch_learnable = AdaptivePatchLayer_v4(input_length=input_length, num_channels=num_node,
                                                             d=patch_dim)
        num_patch = num_patch_max
        self.num_node = num_node
        self.tem_map = tem_map



        self.nonlin_map = Feature_extractor_1DCNN_REG(1, lstmhidden_dim, lstmout_dim,kernel_size=conv_kernel)
        if tem_map:
            self.temporal_mapping = nn.Linear(input_length, int(input_length/2))
            Conv_out = int(input_length/2)-conv_kernel+3
        else:
            Conv_out = input_length - conv_kernel + 3
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim*Conv_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2*hidden_dim,0.1,max_len=5000)

        self.MPNN1 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, num_patch, time_window_size=window_kernel[0], stride=stride[0], decay = decay, pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, num_patch, time_window_size=window_kernel[1], stride=stride[1], decay = decay, pool_choice=pooling_choice)
        num_windows = int((num_patch-window_kernel[0])/stride[0])+int((num_patch-window_kernel[1])/stride[1])+2
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_dim * num_windows * num_node, 2*hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2*hidden_dim, 2*hidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2*hidden_dim, hidden_dim)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(hidden_dim, n_class)),

        ]))



    def forward(self, X):
        if X.size(2) == self.num_node:
            X = X.transpose(-1,-2)

        X = self.adap_Patch(X)
        X = self.num_patch_learnable(X)
        bs, tlen, num_node, dimension = X.size()

        ### Graph Generation
        if self.tem_map:
            A_input = tr.reshape(X, [bs * tlen * num_node, dimension])
            A_input = self.temporal_mapping(A_input)
            A_input = tr.reshape(A_input, [bs * tlen * num_node, A_input.size(-1), 1])
        else:
            A_input = tr.reshape(X, [bs * tlen * num_node, dimension, 1])

        A_input_ = self.nonlin_map(A_input)
        A_input_ = tr.reshape(A_input_, [bs*tlen*num_node,-1])
        A_input_ = self.nonlin_map2(A_input_)
        A_input_ = tr.reshape(A_input_, [bs, tlen,num_node,-1])

        # print('A_input size is ', A_input_.size())

        ## positional encoding before mapping starting
        X_ = tr.reshape(A_input_, [bs,tlen,num_node, -1])
        X_ = tr.transpose(X_,1,2)
        X_ = tr.reshape(X_,[bs*num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = tr.reshape(X_,[bs,num_node, tlen, -1])
        X_ = tr.transpose(X_,1,2)
        A_input_ = X_


        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)


        features1 = tr.reshape(MPNN_output1, [bs, -1])
        features2 = tr.reshape(MPNN_output2, [bs, -1])

        features = tr.cat([features1,features2],-1)

        features = self.fc(features)

        return features


class GAP_STGNN_CLS(nn.Module):
    def __init__(self, lstmhidden_dim, lstmout_dim, conv_kernel, hidden_dim, num_patch_max, input_length, num_node, window_kernel,stride,decay, pooling_choice, n_class, tem_map, patch_dim):
        super(GAP_STGNN_CLS, self).__init__()
        self.adap_Patch = BinaryGaussianPatchLayer_v3(num_patch_max, input_length=input_length,
                                                          num_channels=num_node)
        self.num_patch_learnable = AdaptivePatchLayer_v4(input_length=input_length, num_channels=num_node,
                                                             d=patch_dim)
        num_patch = num_patch_max
        self.num_node = num_node
        self.tem_map = tem_map


        self.nonlin_map = Feature_extractor_1DCNN_CLS(1, lstmhidden_dim, lstmout_dim,kernel_size=conv_kernel)
        if tem_map:
            self.temporal_mapping = nn.Linear(input_length, int(input_length / 2))
            Conv_out = calculate_output_length(int(input_length / 2),self.nonlin_map)
        else:
            Conv_out = calculate_output_length(input_length,self.nonlin_map)
        self.nonlin_map2 = nn.Sequential(
            nn.Linear(lstmout_dim*Conv_out, 2*hidden_dim),
            nn.BatchNorm1d(2*hidden_dim)
        )

        self.positional_encoding = PositionalEncoding(2*hidden_dim,0.1,max_len=5000)


        self.MPNN1 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, num_patch, time_window_size=window_kernel[0], stride=stride[0], decay = decay, pool_choice=pooling_choice)
        self.MPNN2 = GraphConvpoolMPNN_block_v6(2*hidden_dim, hidden_dim, num_node, num_patch, time_window_size=window_kernel[1], stride=stride[1], decay = decay, pool_choice=pooling_choice)

        num_windows = int((num_patch - window_kernel[0]) / stride[0]) + int(
            (num_patch - window_kernel[1]) / stride[1]) + 2
        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(hidden_dim * num_windows * num_node, 2*hidden_dim)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2*hidden_dim, 2*hidden_dim)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2*hidden_dim, hidden_dim)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(hidden_dim, n_class)),

        ]))



    def forward(self, X):

        if X.size(2) == self.num_node:
            X = X.transpose(-1,-2)
        X = self.adap_Patch(X)
        X = self.num_patch_learnable(X)
        bs, tlen, num_node, dimension = X.size() ### tlen = 1

        ### Graph Generation

        if self.tem_map:
            A_input = tr.reshape(X, [bs * tlen * num_node, dimension])
            A_input = self.temporal_mapping(A_input)
            A_input = tr.reshape(A_input, [bs * tlen * num_node, A_input.size(-1), 1])
        else:
            A_input = tr.reshape(X, [bs * tlen * num_node, dimension, 1])


        A_input_ = self.nonlin_map(A_input)
        A_input_ = tr.reshape(A_input_, [bs*tlen*num_node,-1])
        A_input_ = self.nonlin_map2(A_input_)

        A_input_ = tr.reshape(A_input_, [bs, tlen,num_node,-1])

        # print('A_input size is ', A_input_.size())

        ## positional encoding before mapping starting
        X_ = tr.reshape(A_input_, [bs,tlen,num_node, -1])
        X_ = tr.transpose(X_,1,2)
        X_ = tr.reshape(X_,[bs*num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = tr.reshape(X_,[bs,num_node, tlen, -1])
        X_ = tr.transpose(X_,1,2)
        A_input_ = X_

        ## positional encoding before mapping ending

        MPNN_output1 = self.MPNN1(A_input_)
        MPNN_output2 = self.MPNN2(A_input_)


        features1 = tr.reshape(MPNN_output1, [bs, -1])
        features2 = tr.reshape(MPNN_output2, [bs, -1])

        features = tr.cat([features1,features2],-1)

        features = self.fc(features)

        return features
