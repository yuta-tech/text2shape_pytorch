import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.autograd import grad
from torch.autograd import Variable

def hinge_loss(logits, g_d):
    assert g_d=='gen' or g_d=='dis_real' or g_d=='dis_fake', 'arg "g_d" must be "gen" or "dis_read" or "dis_fake"'
    if g_d == 'gen':
        return -torch.mean(logits)
    elif g_d == 'dis_real':
        return torch.mean(F.relu(1. - logits))
    elif g_d == 'dis_fake':
        return torch.mean(F.relu(1. + logits))


def wasserstein_loss(logits, g_d):
    assert g_d=='gen' or g_d=='dis_real' or g_d=='dis_fake', 'arg "g_d" must be "gen" or "dis_read" or "dis_fake"'
    if g_d == 'gen':
        return -torch.mean(logits)
    elif g_d == 'dis_real':
        return -torch.mean(logits)
    elif g_d == 'dis_fake':
        return torch.mean(logits)


    
    
def discriminaotr_loss(netD, real_voxel, fake_voxel, mat_emb, mis_emb, real_labels, fake_labels, loss_type):
    # get 4*4 feature
    real_feature = netD(real_voxel)
    fake_feature = netD(fake_voxel.detach())
    # conditional loss
    # match real
    cond_mat_real_prob, cond_mat_real_logits = netD.COND_DNET(real_feature, mat_emb)
    # mismatch real
    cond_mis_real_prob, cond_mis_real_logits = netD.COND_DNET(real_feature, mis_emb)
    # fake
    cond_fake_prob, cond_fake_logits = netD.COND_DNET(fake_feature, mat_emb)
    
    # losses
    if loss_type == 'BCE':
        cond_d_mat_real_loss = nn.BCELoss()(cond_mat_real_prob.view(-1, 1), real_labels)
        cond_d_mis_real_loss = nn.BCELoss()(cond_mis_real_prob.view(-1, 1), fake_labels)
        cond_d_fake_loss = nn.BCELoss()(cond_fake_prob.view(-1, 1), fake_labels)
    elif loss_type == 'hinge':
        # refer to https://arxiv.org/pdf/1605.05396.pdf
        cond_d_mat_real_loss = hinge_loss(cond_mat_real_logits, 'dis_real')
        cond_d_mis_real_loss = hinge_loss(cond_mis_real_logits, 'dis_fake')
        cond_d_fake_loss = hinge_loss(cond_fake_logits, 'dis_fake')
    
    
    
    # unconditional losss
    # match real
    uncond_mat_real_prob, uncond_mat_real_logits = netD.UNCOND_DNET(real_feature)
    # mismatch real
    uncond_mis_real_prob, uncond_mis_real_logits = netD.UNCOND_DNET(real_feature)
    # fake
    uncond_fake_prob, uncond_fake_logits = netD.UNCOND_DNET(fake_feature)
    
    # losses
    if loss_type == 'BCE':
        uncond_d_mat_real_loss = nn.BCELoss()(uncond_mat_real_prob.view(-1, 1), real_labels)
        uncond_d_fake_loss = nn.BCELoss()(uncond_fake_prob.view(-1, 1), fake_labels)
    elif loss_type == 'hinge':
        # refer to https://arxiv.org/pdf/1605.05396.pdf
        uncond_d_mat_real_loss = hinge_loss(uncond_mat_real_logits, 'dis_real')
        uncond_d_fake_loss = hinge_loss(uncond_fake_logits, 'dis_fake')
    
    
    # total loss
    if loss_type == 'BCE':
        d_loss = ((cond_d_mat_real_loss + uncond_d_mat_real_loss)/2. +
                 (cond_d_fake_loss + uncond_d_fake_loss + cond_d_mis_real_loss)/3. )
    elif loss_type == 'hinge':
         # TO MODIFY
        d_loss = ((cond_d_mat_real_loss + uncond_d_mat_real_loss)/2. +
                 (cond_d_fake_loss + uncond_d_fake_loss + cond_d_mis_real_loss)/3. )
    return d_loss
    
    
def generator_loss(netsD, fake_voxel, mat_emb, real_labels, fake_labels, loss_type):
    total_g_loss = 0.
    for i, netD in enumerate(netsD):
        features = netD(fake_voxel[i])
        cond_prob, cond_logits = netD.COND_DNET(features, mat_emb)
        uncond_prob, uncond_real_logits = netD.UNCOND_DNET(features)
        if loss_type == 'BCE':
            cond_g_loss = nn.BCELoss()(cond_prob.view(-1, 1), real_labels)
            uncond_g_loss = nn.BCELoss()(uncond_prob.view(-1, 1), real_labels)
            g_loss = cond_g_loss + uncond_g_loss
        elif loss_type == 'hinge':
            cond_g_loss = hinge_loss(cond_logits, 'gen')
            uncond_g_loss = hinge_loss(uncond_logits, 'gen')
            g_loss = cond_g_loss + uncond_g_loss
            
        total_g_loss += g_loss
    return total_g_loss
        
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD       
        

def gradient_penalty(fake_shape, real_shape, fake_emb, real_emb, discriminator, device):
    # this function require outputs through sigmoid!
    gp_coff = 10
    alpha = float(torch.randint(2, (1,)))
    
    shape = (alpha*fake_shape + (1.-alpha)*real_shape)
    emb = (alpha*fake_emb + (1.-alpha)*real_emb)
    
    shape = Variable(shape, requires_grad=True).to(device)
    emb = Variable(emb, requires_grad=True).to(device)
    
    _, output = discriminator(shape, emb)
    
#     print(output.size())
    
    grad_shape, grad_emb = grad(
        outputs=output,
        inputs=[shape, emb],
        grad_outputs=torch.ones(output.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)
    
    grad_shape = grad_shape.view(real_shape.size(0), -1)
    
    # torch 1.4.0 doesn't have torch.square ;(
    slopes_shape = torch.sqrt(torch.sum(torch.mul(grad_shape, grad_shape), dim=1))
    slopes_emb = torch.sqrt(torch.sum(torch.mul(grad_emb, grad_emb), dim=1))
    
    gp_shape = torch.mean((slopes_shape - 1.)**2)
    gp_emb = torch.mean((slopes_emb - 1.)**2)
    gp = torch.add(gp_shape, gp_emb)
    d_loss_gp = torch.mul(gp, gp_coff)
    
    return d_loss_gp



