import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import cfg



class T2SWDiscriminator(nn.Module):
    def __init__(self, emb_size=128):
        super(T2SWDiscriminator, self).__init__()
        self.emb_size = emb_size
        
        #model definition
        self.conv1 = nn.Conv3d(4, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv3d(512, 256, kernel_size=2, stride=2, padding=1)
        
        self.dense_emb1 = nn.Linear(self.emb_size, 256)
        self.dense_emb2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(2304, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, 1)
        
        # weight init
        nn.init.xavier_uniform_(self.dense_emb1.weight)
        nn.init.xavier_uniform_(self.dense_emb2.weight)
        nn.init.xavier_uniform_(self.dense3.weight)
        nn.init.xavier_uniform_(self.dense4.weight)
        nn.init.xavier_uniform_(self.dense5.weight)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.xavier_uniform_(self.conv5.weight)
        
        
    def forward(self, shape, emb):
        # shape conv
        # channnel first
        x = self.conv1(shape)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv3(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv4(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv5(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        
        
        # embedding text fc
        y = self.dense_emb1(emb)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = self.dense_emb2(y)
        y = F.leaky_relu(y, negative_slope=0.2)
        
        
        # concat
        x = x.view(x.size()[0], -1)
#         return x, y
        concat = torch.cat((x, y), dim=1)
        # fc layer
        fc = self.dense3(concat)
        fc = F.leaky_relu(fc, negative_slope=0.2)
        fc = self.dense4(fc)
        fc = F.leaky_relu(fc, negative_slope=0.2)
        
        logits = self.dense5(fc)
        
        sigmoid_out = torch.sigmoid(logits)
        return sigmoid_out, logits

    


