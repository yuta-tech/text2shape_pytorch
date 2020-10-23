import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from torchsummary import summary

from discriminator import T2SWDiscriminator
from generator import T2SWGenerator, T2SWGenerator2
from utils.config import cfg

import utils.losses as losses
import dataset as dataset

from utils.util_fnc import weights_init

import numpy as np
import os
import os.path
import datetime




class Trainer():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        
        self.decay_step = cfg.TRAIN.DECAY_STEP
        self.cpkt_step = cfg.TRAIN.CPKT_STEP
        self.save_voxel_step = cfg.TRAIN.SAVE_VOXEL_STEP
        self.print_step = cfg.TRAIN.PRINT_STEP
        self.dstep = 5
        
        self.epochs = cfg.TRAIN.EPOCHS
        self.loss_type = cfg.TRAIN.LOSS
        self.noise_size = cfg.GAN.Z_DIM
        self.noise_unif_abs_max = cfg.GAN.NOISE_UNIF_ABS_MAX
        self.logdir = cfg.TRAIN.LOGDIR
        
    
    def build_model(self):
        # Generator
#         netG = T2SWGenerator2().to(self.device)
        netG = T2SWGenerator().to(self.device)
        # Discriminaotrs
        netD = T2SWDiscriminator().to(self.device)
        
        return netG, netD
        
    def define_optimizers(self, netG, netD):
        optimizerD = optim.Adam(netD.parameters(), lr=cfg.TRAIN.D_LR, betas=(0.0, 0.9))
        optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.G_LR, betas=(0.0, 0.9))
        return optimizerG, optimizerD
        
    def set_logdir(self):
        if not os.path.isdir(self.logdir):
            os.makedirs(self.logdir)
        
        # mkdir datetime name dir
        now = datetime.datetime.now()
        self.dtdir = f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}/'    
        
        self.vizdir = self.logdir + self.dtdir + 'viz/'
        if not os.path.isdir(self.vizdir):
            os.makedirs(self.vizdir)

        self.ckpdir = self.logdir + self.dtdir + 'checkpoints/'
        if not os.path.isdir(self.ckpdir):
            os.makedirs(self.ckpdir)
    
    def save_cpkt(self, epoch, step, optimizerG, optimizerD, netG, netD, d_loss, g_loss):
        
        arg_dict = {
            'epoch' : epoch,
            'step' : step,
            'optimizer_state_dict_G' : optimizerG.state_dict(),
            'model_state_dict_G' : netG.state_dict(),
            'optimizer_state_dict_D' : optimizerD.state_dict(),
            'model_state_dict_D' : netD.state_dict(),
            'd_loss' : d_loss,
            'g_loss' : g_loss,
        }
        
        torch.save(arg_dict, self.ckpdir+f'e{epoch}_s{step}_d{d_loss}_g{g_loss}.pth')
        print('checkpoint is saved')
    
    def save_voxel(self, fake_voxel, real_voxel, step):
        dataset.write_nrrd(fake_voxel[0].to('cpu').detach().numpy().copy(), self.vizdir+f'{step}_generated.nrrd')
        dataset.write_nrrd(real_voxel[0].to('cpu').detach().numpy().copy(), self.vizdir+f'{step}_truth.nrrd')
        print('generated voxel is saved')
    
    def model_summary(self, netG, netD):
        summary(netG, [(8,),(128,)], 8)
        summary(netD, [(4,32,32,32), (128,)], 8)
    
    def train(self, summary=False):
        
        # build model
        netG, netD = self.build_model()
        
        # just summary model
        if summary:
            self.model_summary(netG, netD)
            return
        
        # define optimizerss
        optimizerG, optimizerD = self.define_optimizers(netG, netD)
        scheduler_G = ExponentialLR(optimizerG, gamma=0.95)
        # create log directory
        self.set_logdir()
        # tensorboard
        writer = SummaryWriter()
        
        total_d_loss = 0.
        
        step = 1
        for epoch in range(self.epochs):
            # train
            netD.train()
            netG.train()
            for i, data in enumerate(self.dataloader):                
                fake, mat_real, mis_real = data[0], data[1], data[2]
                
                true_voxel, fake_embedding = fake[0].to(self.device, dtype=torch.float), fake[1].to(self.device, dtype=torch.float)
                
                mat_voxel, mat_embedding = mat_real[0].to(self.device, dtype=torch.float), mat_real[1].to(self.device, dtype=torch.float)
                
                mis_voxel, mis_embedding = mis_real[0].to(self.device, dtype=torch.float), mis_real[1].to(self.device, dtype=torch.float)


                # train Discriminator
                netD.zero_grad()
                
                # get G output
                noise = torch.from_numpy(np.random.uniform(low=-self.noise_unif_abs_max, high=self.noise_unif_abs_max, size=[mat_embedding.size(0), self.noise_size])).to(self.device, dtype=torch.float)
                    
                fake_voxel = netG(noise, fake_embedding)
                
                
                _, fake_logits = netD(fake_voxel.detach(), fake_embedding)
                _, mat_real_logits = netD(mat_voxel, mat_embedding)
                _, mis_real_logits =netD(mis_voxel, mis_embedding)
                d_loss = losses.wasserstein_loss(fake_logits, 'dis_fake') + 2.0*losses.wasserstein_loss(mat_real_logits, 'dis_real') +  losses.wasserstein_loss(mis_real_logits, 'dis_fake')
                d_gp = losses.gradient_penalty(fake_voxel, mat_voxel, fake_embedding, mat_embedding, netD, self.device)
                d_loss += d_gp
                
                total_d_loss = total_d_loss + d_loss.item()
                d_loss.backward()
                optimizerD.step()

                # train G
                if step % self.dstep == 0:
#                     print('g time', step)
                    netG.zero_grad()
                    _, fake_logits = netD(fake_voxel,fake_embedding)
                    g_loss = losses.wasserstein_loss(fake_logits, 'gen')
                    
                    
                    g_loss.backward()
                    optimizerG.step()
                    
                    # tensorboard
                    writer.add_scalar('d_loss/train', d_loss.item(), step//self.dstep)
                    writer.add_scalar('d_gp/train', d_gp.item(), step//self.dstep)
                    writer.add_scalar('g_loss/train',g_loss.item(), step//self.dstep)
                
                if step % (self.decay_step*self.dstep) == 0:
                    print('g_lr is decay!')
                    scheduler_G.step()
                
                if step % (self.print_step*self.dstep) == 0:
                    print('global step:',step)
                    print(f'train--->step:{step//self.dstep}')
                    print(f'd_loss:{d_loss.item()}')
                    print(f'd_gp:{d_gp.item()}')
                    print(f'g_loss:{g_loss.item()}')
                    
                    
                
                # checkpoint
                if (step % self.cpkt_step*self.dstep) == 0:
                    self.save_cpkt(epoch, step, optimizerG, optimizerD, netG, netD, d_loss, g_loss)
                
                # generated shape save
                if step % (self.save_voxel_step*self.dstep) == 0:
#                     print(fake_voxel.size())
                    self.save_voxel(fake_voxel, true_voxel, step//self.dstep)
                
                step = step + 1