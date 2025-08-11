"""
This file define the deep learning model for agents.
"""
# Basic lib
import os
import numpy as np
import time
from tqdm import tqdm
import random

# Neural network lib
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
torch.autograd.set_detect_anomaly(True)

class MAADAgent(nn.Module):
    def __init__(self, error_types: int = 42, corresponds_svc: str = "", rca_types: int = 5) -> None:
        super().__init__()
        self.corresponds_svc = corresponds_svc
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool2 = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(1)

        self.feature_compress = nn.LSTM(303, 303, 2)
        self.category = nn.Linear(303, error_types)

        self.rca = nn.Linear(303, rca_types) # 0: normal    1: return   2: exception    3: cpu_contention   4: network_delay


    def forward(self, input):
        # Feature fusion
        input = self.norm(input)

        output = self.conv1(input=input)
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        output = self.conv3(output)
        output = self.pool3(output)
        output = self.norm(output)
        output = output.squeeze(0)
        output, _ = self.feature_compress(output)
        output = output[:, -1, :]
        classification = self.category(output)
        classification = torch.softmax(classification, dim=-1)

        rca = torch.softmax(self.rca(output), dim=-1)
        # classification = nn.functional.adaptive_avg_pool2d(output, output_size=(1,15))
        return self.corresponds_svc, output.unsqueeze(0).unsqueeze(0), classification, rca