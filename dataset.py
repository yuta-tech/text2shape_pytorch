import torch
from torch.utils.data import Dataset

import numpy as np
import nrrd
import pickle
import random
import copy
import math
from utils.config import cfg

from sklearn.model_selection import train_test_split

def write_nrrd(voxel_tensor, filename):
    """Converts binvox tensor to NRRD (RGBA) format and writes it if a filename is provided.
    Args:
        voxel_tensor: A tensor representing the binary voxels. Values can range from 0 to 1, and
            they will be properly scaled.
        filename: Filename that the NRRD will be written to.
    Writes:
        nrrd_tensor: An RGBA tensor where the channel dimension (RGBA) comes first
            (channels, height, width, depth).
    """
    if voxel_tensor.ndim == 3:  # Add a channel if there is no channel dimension
        voxel_tensor = voxel_tensor[np.newaxis, :]
    elif voxel_tensor.ndim == 4:  # Roll axes so order is (channel, height, width, depth) (not sure if (h, w, d))
#         voxel_tensor = np.rollaxis(voxel_tensor, 3)
        # pytorch is channel first
        pass
    else:
        raise ValueError('Voxel tensor must have 3 or 4 dimensions.')

    # Convert voxel_tensor to uint8
    voxel_tensor = (voxel_tensor * 255).astype(np.uint8)
#     voxel_tensor = ((voxel_tensor + 1.) * 128).astype(np.uint8)
    voxel_tensor = np.clip(voxel_tensor, 0, 255).astype(np.uint8)
    if voxel_tensor.shape[0] == 1:  # Add channels if voxel_tensor is a binvox tensor
        nrrd_tensor_slice = voxel_tensor
        nrrd_tensor = np.vstack([nrrd_tensor_slice] * 4)
        nrrd_tensor[:3, :, :, :] = 128  # Make voxels gray
        nrrd_tensor = nrrd_tensor.astype(np.uint8)
    elif voxel_tensor.shape[0] == 4:
        nrrd_tensor = voxel_tensor
    elif voxel_tensor.shape[0] != 4:
        raise ValueError('Voxel tensor must be single-channel or 4-channel')

    nrrd.write(filename, nrrd_tensor)

def read_nrrd(file_name):
    nrrd_voxel, options = nrrd.read(file_name)
    assert nrrd_voxel.ndim == 4
    
    # 0. to 1.
    voxel = nrrd_voxel.astype(np.float64)/255.
    # -1 to 1
#     voxel = nrrd_voxel.astype(np.float64)/128. - 1.
    
    # モデルを立たせる
    voxel = np.swapaxes(voxel, 1, 2)
    voxel = np.swapaxes(voxel, 1, 3)
    return voxel

def open_pickle(pickle_file):
    with open(pickle_file, 'rb') as pk:
        input_dict = pickle.load(pk)
        return input_dict

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def augment_voxel(voxel, max_noise=10):
    # augment rgb value by adding noise
    augmented_voxel = np.copy(voxel)
    if (voxel.ndim == 4) and (voxel.shape[0] != 1) and (max_noise > 0):
        noise_val = float(np.random.randint(-max_noise, high=(max_noise + 1))) / 255
#         noise_val = float(np.random.randint(-max_noise, high=(max_noise + 1))) / 128. - 1.
        augmented_voxel[:3, :, :, :] += noise_val
        augmented_voxel = np.clip(augmented_voxel, 0., 1.)
#     augmented_voxel = np.clip(augmented_voxel, -1., 1.)
    return augmented_voxel
    
# not needed
def _remove_bad_model(caption_tuples):
    bad_models = open_pickle(cfg.DIR.DATA_PATH + 'shapenet/problematic_nrrds_shapenet_unverified_256_filtered_div_with_err_textures.p')
    
    # search bad models
    bad_idxs = []
    for i, caption_tuple in enumerate(caption_tuples):
        if caption_tuple[2] in bad_models:
            bad_idxs.append(i)
    
    for i in range(len(bad_idxs)-1, 0, -1):
        caption_tuples.pop(bad_idx[i])
    
    return caption_tuples

def resplit_sample_train_val(train_split, val_split, seed):
    train_embedding_pickle = cfg.DIR.DATA_PATH + 'shapenet/shapenet-embeddings/text_embeddings_train.p'
    val_embedding_pickle = cfg.DIR.DATA_PATH + 'shapenet/shapenet-embeddings/text_embeddings_val.p'
    
    train_caption_tuples = open_pickle(train_embedding_pickle)['caption_embedding_tuples']
    val_caption_tuples = open_pickle(val_embedding_pickle)['caption_embedding_tuples']
    
    # concat
    all_caption_tuples = train_caption_tuples + val_caption_tuples
    
    # resplit
    train_caption_tuples, val_caption_tuples = train_test_split(all_caption_tuples, test_size=val_split, shuffle=False)
    
    save_pickle(train_caption_tuples, cfg.DIR.DATA_PATH + 'shapenet/shapenet-embeddings/resplit_caption_tuples_train.p')
    save_pickle(val_caption_tuples, cfg.DIR.DATA_PATH + 'shapenet/shapenet-embeddings/resplit_caption_tuples_val.p')

def resplit_sample(train_split, val_split, test_split, seed):
    train_embedding_pickle = cfg.DIR.DATA_PATH + 'shapenet/shapenet-embeddings/text_embeddings_train.p'
    val_embedding_pickle = cfg.DIR.DATA_PATH + 'shapenet/shapenet-embeddings/text_embeddings_val.p'
    test_embedding_pickle = cfg.DIR.DATA_PATH + 'shapenet/shapenet-embeddings/text_embeddings_test.p'
    
    train_caption_tuples = open_pickle(train_embedding_pickle)['caption_embedding_tuples']
    val_caption_tuples = open_pickle(val_embedding_pickle)['caption_embedding_tuples']
    test_caption_tuples = open_pickle(test_embedding_pickle)['caption_embedding_tuples']
    
    # concat
    all_caption_tuples = train_caption_tuples + val_caption_tuples + test_caption_tuples
    
    # resplit
    train_caption_tuples, test_caption_tuples = train_test_split(all_caption_tuples, test_size=test_split, shuffle=False)
    train_caption_tuples, val_caption_tuples = train_test_split(train_caption_tuples, test_size=val_split/(train_split+val_split), shuffle=False)
    
    save_pickle(train_caption_tuples, cfg.DIR.DATA_PATH + 'shapenet/shapenet-embeddings/resplit_caption_tuples_train.p')
    save_pickle(val_caption_tuples, cfg.DIR.DATA_PATH + 'shapenet/shapenet-embeddings/resplit_caption_tuples_val.p')
    save_pickle(test_caption_tuples, cfg.DIR.DATA_PATH + 'shapenet/shapenet-embeddings/resplit_caption_tuples_test.p')

class ShapeNetDataset(Dataset):
    def __init__(self, mode, resplit=False):
        super().__init__()
        assert mode in ['train', 'val', 'test']
        
        if not resplit:
            self.embedding_pickle = cfg.DIR.DATA_PATH + f'shapenet/shapenet-embeddings/text_embeddings_{mode}.p'
            # embedding_text and model tuple
            self.caption_tuples = open_pickle(self.embedding_pickle)['caption_embedding_tuples']
            # index list for mismatch text extracting
            self.index_list = [x for x in range(len(self.caption_tuples))]
        else:
            if mode in ['train', 'val']:
                resplit_pickle = cfg.DIR.DATA_PATH + f'shapenet/shapenet-embeddings/resplit_caption_tuples_{mode}.p'
                self.caption_tuples = open_pickle(resplit_pickle)
            else:
                self.embedding_pickle = cfg.DIR.DATA_PATH + f'shapenet/shapenet-embeddings/text_embeddings_{mode}.p'
                # embedding_text and model tuple
                self.caption_tuples = open_pickle(self.embedding_pickle)['caption_embedding_tuples']
            # index list for mismatch text extracting
            self.index_list = [x for x in range(len(self.caption_tuples))]
        self.shapenet_dir = cfg.DIR.RGB_VOXEL_PATH + 'nrrd_256_filter_div_32_solid'
        
        self.fake_index = np.random.permutation(self.index_list)
        self.mis_real_index = np.random.permutation(self.index_list)
        self.mat_real_index = np.random.permutation(self.index_list)
        
    def load_voxel(self, category, model_id):
        # return nrrd file
        # shapenet format
        return read_nrrd(self.shapenet_dir+f'/{model_id}/{model_id}.nrrd')
    
    def get_data(self, choice, idx):
        assert choice in ['fake', 'mis', 'mat']
        if choice == 'mis': 
            # shape loading
            caption_tuple = self.caption_tuples[idx]
            category = caption_tuple[1]
            model_id = caption_tuple[2]
            voxel = self.load_voxel(category, model_id)
            augmented_voxel = torch.from_numpy(augment_voxel(voxel))
    
            # match text
            learned_embedding = torch.from_numpy(caption_tuple[3])
            # mismatch text
            ilist = copy.deepcopy(self.index_list)
            ilist.remove(idx)
            while True:
                # pick up mismatch sample
                mismatch_idx = ilist[random.randint(0, len(ilist)-1)]
                mismatch_model_id = self.caption_tuples[mismatch_idx][2]
                
                if mismatch_model_id != model_id:
                    # get mismatch sample
                    mismatch_learned_embedding = torch.from_numpy(self.caption_tuples[mismatch_idx][3])
                    break
                else:
                    # get same model
                    # One model has five caption
                    ilist.remove(mismatch_idx)
            
            return augmented_voxel, mismatch_learned_embedding
        elif choice == 'mat':
            # shape loading
            caption_tuple = self.caption_tuples[idx]
            category = caption_tuple[1]
            model_id = caption_tuple[2]
            voxel = self.load_voxel(category, model_id)
            augmented_voxel = torch.from_numpy(augment_voxel(voxel))
    
            # match text
            match_learned_embedding = torch.from_numpy(caption_tuple[3])
            
            return augmented_voxel, match_learned_embedding
        elif choice == 'fake':
            ## shape loading
            caption_tuple = self.caption_tuples[idx]
            category = caption_tuple[1]
            model_id = caption_tuple[2]
            voxel = self.load_voxel(category, model_id)
            augmented_voxel = torch.from_numpy(augment_voxel(voxel))
    
            # match text
            learned_embedding = torch.from_numpy(caption_tuple[3])
            
            return augmented_voxel, learned_embedding
    
    def __len__(self):
        return len(self.caption_tuples)
    
    def __getitem__(self, idx):
        fake = self.get_data('fake', self.fake_index[idx])
        mat_real = self.get_data('mat', self.mat_real_index[idx])
        mis_real = self.get_data('mis', self.mis_real_index[idx])
        return fake, mat_real, mis_real


class TestShapeNetDataset(Dataset):
    def __init__(self, resplit=False):
        super().__init__()

        self.embedding_pickle = cfg.DIR.DATA_PATH + f'shapenet/shapenet-embeddings/text_embeddings_test.p'
        # embedding_text and model tuple
        self.caption_tuples = open_pickle(self.embedding_pickle)['caption_embedding_tuples']
        # index list for mismatch text extracting
        self.index_list = [x for x in range(len(self.caption_tuples))]
        

        self.shapenet_dir = cfg.DIR.RGB_VOXEL_PATH + 'reso32/nrrd_256_filter_div_32_solid'

            

        
    def load_voxel(self, category, model_id):
        # return nrrd file
        # shapenet format
        return read_nrrd(self.shapenet_dir+f'/{model_id}/{model_id}.nrrd')

        return voxels
        
    def __len__(self):
        return len(self.caption_tuples)
    
    def __getitem__(self, idx):
        # shape loading
        caption_tuple = self.caption_tuples[idx]
        category = caption_tuple[1]
        model_id = caption_tuple[2]
        voxel = self.load_voxel(category, model_id)
        augmented_voxel = torch.from_numpy(augment_voxel(voxel))

        # match text
        learned_embedding = torch.from_numpy(caption_tuple[3])
#         # mismatch text
#         ilist = copy.deepcopy(self.index_list)
#         ilist.remove(idx)
#         while True:
#             # pick up mismatch sample
#             mismatch_idx = ilist[random.randint(0, len(ilist)-1)]
#             mistmatch_model_id = self.caption_tuples[mismatch_idx][2]
            
#             if mistmatch_model_id != model_id:
#                 # get mismatch sample
#                 mistmatch_learned_embedding = torch.from_numpy(self.caption_tuples[mismatch_idx][3])
#                 break
#             else:
#                 # get same model
#                 # One model has five caption
#                 ilist.remove(mismatch_idx)
        
        return learned_embedding, augmented_voxel


# test code
# from torch.utils.data import DataLoader
# shapenet_dataset = ShapeNetDataset(mode='train')
# dataloader = DataLoader(shapenet_dataset, batch_size=8, shuffle=False)
# print(len(shapenet_dataset))

# import os
# li = os.listdir('/home/fukatsu/dataset/shapenet/voxel/reso32/nrrd_256_filter_div_32_solid')
# print(len(li))


# data = iter(dataloader).next()
# print(torch.min(data[1]), torch.max(data[1]))