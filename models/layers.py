import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
quoted from
    https://github.com/ajbrock/BigGAN-PyTorch/blob/master/layers.py
"""

def proj(x, y):
    return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x

def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                 u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs

class SN(object):
    def __init__(self, num_svs, num_itrs, weight_dim_zero, transpose=False, eps=1e-12):
        """
            weight_dim_zero : weight shape (dim0, dim1, shape_size) -> dim0 is
        """
        # Number of power iterations per step
        self.num_itrs = num_itrs
        # Number of singular values
        self.num_svs = num_svs
        # Transposed?
        self.transpose = transpose
        # Epsilon value for avoiding divide-by-0
        self.eps = eps
        # Register a singular vector for each sv
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, weight_dim_zero))
            self.register_buffer('sv%d' % i, torch.ones(1))
    
    # Singular vectors (u side)
    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    # Singular values; 
    # note that these buffers are just for logging and are not used in training. 
    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
     
    # Compute the spectrally-normalized weight
    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        # Apply num_itrs power iterations
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
        # Update the svs
        if self.training:
            with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv     
        return self.weight / svs[0]

# modified
# 3D Conv layer with spectral norm
class SNConv3d(nn.Conv3d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, 
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv3d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, 
                         padding=padding, dilation=dilation, groups=groups, bias=bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)
        
    def forward(self, x):
        return F.conv3d(x, self.W_(), bias=self.bias, stride=self.stride, 
                      padding=self.padding, dilation=self.dilation, groups=self.groups)
"""
二重継承でconvが持つパラーメタをSNで更新？して最終的にfunctionalに渡してる
"""
# new created
class SNConvTranspose3d(nn.ConvTranspose3d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, output_padding=0, groups=1, bias=True, 
                 num_svs=1, num_itrs=1, eps=1e-12):
        nn.ConvTranspose3d.__init__(self, in_channels, out_channels, kernel_size, stride=stride, 
                         padding=padding, dilation=dilation, output_padding=output_padding , groups=groups, bias=bias)
        # transpose conv wieght'shape is (in_channel, out_channel, depth, height, width)
        SN.__init__(self, num_svs, num_itrs, in_channels, eps=eps)
        
    def forward(self, x):
        return F.conv_transpose3d(x, self.W_(), bias=self.bias, stride=self.stride, 
                      padding=self.padding, dilation=self.dilation, groups=self.groups, output_padding=self.output_padding)

# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                   num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
        
    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)

class ConditionalBatchNorm3d(nn.Module):
    def __init__(self, emb_size, num_features, eps=1e-4, momentum=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        
        self.bn = nn.BatchNorm3d(num_features, affine=False, eps=eps, momentum=momentum)
#         self.gamma_dense = SNLinear(emb_size, num_features, bias=False)
#         self.beta_dense = SNLinear(emb_size, num_features, bias=False)
        self.gamma_dense = nn.Linear(emb_size, num_features, bias=False)
        self.beta_dense = nn.Linear(emb_size, num_features, bias=False)
        
    def forward(self, inputs, emb):
        bn_out = self.bn(inputs)
        
        gamma = self.gamma_dense(emb)
        beta = self.beta_dense(emb)
        
        out = bn_out*(gamma + 1).view(-1, self.num_features, 1, 1, 1) + beta.view(-1, self.num_features, 1, 1, 1)
        
        return out

class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, emb_size, num_features, eps=1e-4, momentum=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.bn = nn.BatchNorm1d(num_features, affine=False, eps=eps, momentum=momentum)
        

        self.gamma_dense = SNLinear(emb_size, num_features, bias=False)
        self.beta_dense = SNLinear(emb_size, num_features, bias=False)
        
    def forward(self, inputs, emb):
        bn_out = self.bn(inputs)
        
        gamma = self.gamma_dense(emb)
        beta = self.beta_dense(emb)
        
        out = bn_out*(gamma + 1) + beta
        
        return out

