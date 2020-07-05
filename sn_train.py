import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter


import models.generator as generator
import models.discriminator as discriminator

import utils.losses as losses
import utils.dataset as dataset

from torchsummary import summary

import numpy as np

import os
import os.path
import datetime

# cudnn
torch.backends.cudnn.deterministic = True
# random
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# const
batch_size = 8
epochs = 100*(batch_size//8)
D_learning_rate = 1e-4
G_learning_rate = 5e-5
D_timing = 1
D_update_threshold = 1.1
G_timing = 1

# Generator with SN?
SN_G = True
# Generator with cBN?
cbn = True

# resplit dataset?
resplit=True

noise_size = 8
uniform_max = 0.5
noise_unif_abs_max = 1

# decay_steps=10000
decay_steps = 10000

emb_size = 128
txt_noise_size = emb_size + noise_size



# device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model
if SN_G:
    model_G = generator.T2SSNGenerator(emb_size, noise_size, cbn=cbn).to(device)
else:
    model_G = generator.T2SGenerator(emb_size, noise_size, cbn=cbn).to(device)
model_D = discriminator.T2SDiscriminator(emb_size).to(device)

# model summary
# summary(model_G, [(128,),(8,)], 8)
# summary(model_D, [(4,32,32,32), (128,)], 8)


logdir = './log/'
if not os.path.isdir(logdir):
    os.makedirs(logdir)

# mkdir datetime name dir
now = datetime.datetime.now()
dtdir = f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'    

vizdir = logdir + dtdir + 'viz/'
if not os.path.isdir(vizdir):
    os.makedirs(vizdir)

ckpdir = logdir + dtdir + 'checkpoints/'
if not os.path.isdir(ckpdir):
    os.makedirs(ckpdir)

with open(logdir + dtdir + 'sn_frag.txt', 'w') as sn_f:
    if SN_G:
        s = 'SN_G is True'
    else:
        s = 'SN_G is False'
    sn_f.write(s)
    
# save params
param_dict = {
    'seed': seed,
    'batch_size': batch_size,
    'epochs': epochs,
    'D_learning_rate': D_learning_rate,
    'G_learning_rate': G_learning_rate,
    'G_timing': G_timing,
    'SN_G': SN_G,
    'cbn': cbn,
    'noise_size': noise_size,
    'uniform_max': uniform_max,
    'noise_unif_abs_max': noise_unif_abs_max,
    'decay_steps': decay_steps,
    'emb_size': emb_size,
    'resplit': resplit,
}

dataset.save_pickle(param_dict, logdir+dtdir+'params.p')

# tensorboard
writer = SummaryWriter()

# optimizer
optim_G = optim.Adam(model_G.parameters(), lr=G_learning_rate, betas=(0.0, 0.9))
optim_D = optim.Adam(model_D.parameters(), lr=D_learning_rate, betas=(0.0, 0.9))
# optim_D = optim.SGD(model_D.parameters(), lr=D_learning_rate, momentum=0.9)

# learning rate scheduler, use decay step on train loop
scheduler_G = ExponentialLR(optim_G, gamma=0.95)
scheduler_D = ExponentialLR(optim_D, gamma=0.95)



train_shapenet = dataset.ShapeNetDataset(mode='train', resplit=True)
valid_shapenet = dataset.ShapeNetDataset(mode='val', resplit=True)
train_dataloader = DataLoader(train_shapenet, batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_shapenet, batch_size, shuffle=False)

step = 1
d_mat_real_loss_list = []
d_mis_real_loss_list = []
d_fake_loss_list = []
d_loss_list = []
g_loss_list = []

print(f'Using {device}')
print('start learning...')
for epoch in range(epochs):
    # train
    for i, data in enumerate(train_dataloader):
        model_G.train()
        model_D.train()
        """
        optimizer.zero_grad()とmodel.zero_grad()は
        optimizerがmodelのすべてのパラメータを持つならば等しい
        """
        optim_D.zero_grad()
        optim_G.zero_grad()
        
        # train D
        optim_D.zero_grad()
        
        mat_emb, real_voxel, mis_emb = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float), data[2].to(device, dtype=torch.float)
        # match real
        _, mat_real_logits = model_D(real_voxel, mat_emb)
        _, mis_real_logits = model_D(real_voxel, mis_emb)
        # fake
        noise = torch.from_numpy(np.random.uniform(low=-noise_unif_abs_max, high=noise_unif_abs_max, size=[mat_emb.size(0), noise_size])).to(device, dtype=torch.float)
        
        fake_voxel = model_G(mat_emb, noise)
        _, fake_logits = model_D(fake_voxel, mat_emb)
        
        # refer to https://arxiv.org/pdf/1605.05396.pdf
        d_mat_real_loss = losses.hinge_loss(mat_real_logits, 'dis_real')
        d_mis_real_loss = losses.hinge_loss(mis_real_logits, 'dis_fake')
        d_fake_loss = losses.hinge_loss(fake_logits, 'dis_fake')
        
        d_loss = 2*d_mat_real_loss + (d_fake_loss + d_mis_real_loss)
                
        d_loss.backward(retain_graph=True)
        optim_D.step()
        
        d_loss_list.append(d_loss.item())
        d_mat_real_loss_list.append(d_mat_real_loss.item())
        d_mis_real_loss_list.append(d_mis_real_loss.item())
        d_fake_loss_list.append(d_fake_loss.item())
#         if step >= 2:
        if step % G_timing == 0:
            # train G
            optim_G.zero_grad()
            # Get updated D output
            _, fake_logits = model_D(fake_voxel, mat_emb)
            
            g_loss = losses.hinge_loss(fake_logits, 'gen')
            g_loss.backward()
            
            g_loss_list.append(g_loss.item())
            
            optim_G.step()
            
            writer.add_scalar('g_loss/train',g_loss.item(), step)
        
        if step % decay_steps == 0:
            scheduler_G.step()
#             scheduler_D.step()
        
        if step % 20 == 0:
            print(f'train--->step:{step}')
            print(f'd_loss:{d_loss.item()}, d_mat_real_loss:{d_mat_real_loss.item()}, d_mis_real_loss:{d_mis_real_loss.item()}, d_fake_loss:{d_fake_loss.item()}')
            print(f'g_loss:{g_loss.item()}')
        
        # checkpoint
        if step % 1000 == 0:
            torch.save({
                'epoch' : epoch,
                'step' : step,
                'optimizer_state_dict_D' : optim_D.state_dict(),
                'optimizer_state_dict_G' : optim_G.state_dict(),
                'd_loss' : d_loss,
                'g_loss' : g_loss,
            }, ckpdir+f'e{epoch}_s{step}_d{d_loss}_g{g_loss}.pth')
            print('checkpoint is saved')
        
        # generated shape save
        if step % 1000 == 0:
            dataset.write_nrrd(fake_voxel[0].to('cpu').detach().numpy().copy(), vizdir+f'{step}_generated.nrrd')
            print('generated voxel is saved')
        
        
        # tensorboard
        writer.add_scalar('d_loss/train', d_loss.item(), step)
        writer.add_scalar('d_mat_real_loss/train', d_mat_real_loss.item(), step)
        writer.add_scalar('d_mis_real_loss/train', d_mis_real_loss.item(), step)
        writer.add_scalar('d_fake_loss/train', d_fake_loss.item(), step)
        
        step = step + 1
#     validation
#     for i, data in enumerate(valid_dataloader):
    
        

print('end learning...')