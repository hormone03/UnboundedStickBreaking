#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:23:59 2023

@author: akinlolu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import parser, vocab_size
import numpy as np

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class stickBrEncoder(object):
    def __init__(self):
        # encoder

        #self.encoder = nn.Sequential(
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(vocab_size, args.h1_out)
        self.fc2 = nn.Linear(args.h1_out, args.h1_out)

            #F.softplus(nn.Linear(vocab_size, args.h1_out)),
            #F.softplus(nn.Linear(args.h1_out, args.h1_out)),
           # nn.Dropout(0.2),
            #nn.Linear(vocab_size, args.h1_out),
            #nn.ReLU(),
            #nn.Linear(args.h1_out, args.topic_size),
            #nn.Dropout(p=0.2),
        #)
        self.fcmu = nn.Linear(args.h1_out, args.topic_size)
        self.fclv = nn.Linear(args.h1_out, args.topic_size)
        self.bnmu = nn.BatchNorm1d(args.topic_size)
        self.bnlv = nn.BatchNorm1d(args.topic_size)

        #self.h1 = nn.BatchNorm1d(args.topic_size)
        #self.h1.weight.data.copy_(torch.ones(args.topic_size))
        #self.h1.weight.requires_grad = False

        #self.h2 = nn.BatchNorm1d(args.topic_size)
        #self.h2.weight.data.copy_(torch.ones(args.topic_size))
        #self.h2.weight.requires_grad = False

    def encode(self, x):
        h1 = F.softplus(self.fc1(x))
        h2 = F.softplus(self.fc2(h1))
        #return self.drop(h2)
        x_emb = self.drop(h2)
        if args.encoder_output == 1:
            alpha = self.bnmu(self.fcmu(x_emb))
            return alpha #F.softmax(self.h1(x_emb))
        else:
            mu = self.bnmu(self.fcmu(x_emb))
            lv = self.bnlv(self.fclv(x_emb))
            return mu, lv #F.softmax(self.h1(x_emb)), F.softmax(self.h2(x_emb))

class stickBrDecoder(object):
    def __init__(self):

        self.fc = nn.Linear(args.topic_size, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.drop = nn.Dropout(0.2)
        #self.rnn = nn.RNN(input_size=args.topic_size, hidden_size=args.topic_size, num_layers=256)


        #self.decoder = nn.Linear(args.topic_size, vocab_size)
        #self.decoder_norm = nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001, affine=True)
        #self.decoder.weight.data.uniform_(-1.0/np.sqrt(args.topic_size), 1.0/np.sqrt(args.topic_size))
        # remove BN's scale parameters
        #self.decoder_norm.weight.data.copy_(torch.ones(vocab_size))
        #self.decoder_norm.weight.requires_grad = False

    def decode(self, z):
        inputs = self.drop(z)
        #inputs, h_n = self.rnn(inputs) ##  uncomment for stick_brk RNN
        decoder =  F.log_softmax(self.bn(self.fc(inputs)), dim=1) ##  for document modeling: self.bn(self.fc(inputs))
        return decoder