import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
from torch.autograd import Variable

from .layers import ConditionalBatchNorm3d, SNConv3d, SNConvTranspose3d, SNLinear

from .config import cfg

class T2SWGenerator(nn.Module):
    def __init__(self, emb_noise_size=128+8):
        super(T2SWGenerator, self).__init__()
        self.emb_noise_size = emb_noise_size

        #model definition 
        self.dense_in = SNLinear(self.emb_noise_size, 512*4*4*4)
        self.bn1 = nn.BatchNorm1d(512*4*4*4)
        # padding must be 1 !
        self.deconv1 = nn.ConvTranspose3d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(512)
        self.deconv2 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.deconv3 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.deconv4 = nn.ConvTranspose3d(128, 4, kernel_size=4, stride=2, padding=1)
        
        # weight init
        nn.init.xavier_uniform_(self.dense_in.weight)
        nn.init.xavier_uniform_(self.deconv1.weight)
        nn.init.xavier_uniform_(self.deconv2.weight)
        nn.init.xavier_uniform_(self.deconv3.weight)
        nn.init.xavier_uniform_(self.deconv4.weight)
        
        
    def forward(self, noise, emb):
        emb_noise = torch.cat((noise, emb), dim=1)
        x = self.dense_in(emb_noise)
        x = self.bn1(x)
        x = F.relu(x)
        # channnel first
        x = x.view(-1, 512, 4, 4, 4)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.deconv4(x)
        out = torch.sigmoid(x)
#         out = torch.tanh(x)
    
        return out

class T2SWGenerator2(nn.Module):
    def __init__(self, emb_noise_size=128+8):
        super(T2SWGenerator2, self).__init__()
        self.emb_noise_size = emb_noise_size

        #model definition 
        self.dense_in = nn.Linear(self.emb_noise_size, 512*4*4*4)
        self.bn1 = nn.BatchNorm1d(512*4*4*4)
        # padding must be 1 !
        self.deconv1 = nn.ConvTranspose3d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(512)
        self.deconv2 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = ConditionalBatchNorm3d(128, 256)
        self.deconv3 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = ConditionalBatchNorm3d(128, 128)
        self.deconv4 = nn.ConvTranspose3d(128, 4, kernel_size=4, stride=2, padding=1)
        
        # weight init
        nn.init.xavier_uniform_(self.dense_in.weight)
        nn.init.xavier_uniform_(self.deconv1.weight)
        nn.init.xavier_uniform_(self.deconv2.weight)
        nn.init.xavier_uniform_(self.deconv3.weight)
        nn.init.xavier_uniform_(self.deconv4.weight)
        
        
    def forward(self, noise, emb):
        emb_noise = torch.cat((noise, emb), dim=1)
        x = self.dense_in(emb_noise)
        x = self.bn1(x)
        x = F.relu(x)
        # channnel first
        x = x.view(-1, 512, 4, 4, 4)
        x = self.deconv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = self.bn3(x, emb)
        x = F.relu(x)
        x = self.deconv3(x)
        x = self.bn4(x, emb)
        x = F.relu(x)
        x = self.deconv4(x)
        out = torch.sigmoid(x)
#         out = torch.tanh(x)
    
        return out
