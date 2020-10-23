import torch
from torch.utils.data import DataLoader
from models.generator import T2SWGenerator
import utils.dataset as dataset
import numpy as np
from utils.config import cfg


# set random seed, default is 0
# cudnn
torch.backends.cudnn.deterministic = True
# random
seed = cfg.GAN.SEED
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

ckpt_dir = 'scripts/log/6_30_19_24_4/checkpoints/'
test_dir = 'scripts/log/6_30_19_24_4/test/'
checkpoint = torch.load(ckpt_dir+'e59_s55000_d-32.835147857666016_g0.9256608486175537.pth')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# cfg_from_list(['GAN.GF_DIM', 64])
# cfg_from_list(['GAN.DF_DIM', 64])
# cfg_from_list(['GAN.CONDITION_DIM', 128])

# Generator
netG = T2SWGenerator(128+8).to(device)
# # Discriminaotrs
# netsD = []
# if cfg.TREE.BRANCH_NUM > 0:
#     netsD.append(D_NET32().to(self.device))
# if cfg.TREE.BRANCH_NUM > 1:
#     netsD.append(D_NET64().to(self.device))

# for netD in netsD:
#     netD.load_state_dict()
netG.load_state_dict(checkpoint['model_state_dict_G'])
    


shapenet = dataset.TestShapeNetDataset()
test_dataloader = DataLoader(shapenet, batch_size=1, shuffle=False)

# for netD in netsD:
#     netD.eval()
netG.eval()

for i, data in enumerate(test_dataloader):
    with torch.no_grad():
        emb, real_voxel = data[0].to(device, dtype=torch.float), data[1]
        noise = torch.from_numpy(np.random.uniform(low=-cfg.GAN.NOISE_UNIF_ABS_MAX, high=cfg.GAN.NOISE_UNIF_ABS_MAX, size=[emb.size(0), cfg.GAN.Z_DIM])).to(device, dtype=torch.float)
        pred = netG(noise, emb)
        dataset.write_nrrd(pred.view(4,32,32,32).to('cpu').detach().numpy().copy(), test_dir+f'test{i}_generated.nrrd')