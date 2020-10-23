from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.CUDA = True

__C.DIR = edict()
__C.DIR.DATA_PATH = '/home/fukatsu/dataset/text2shape-data/'
__C.DIR.RGB_VOXEL_PATH = '/home/fukatsu/dataset/shapenet/voxel/reso32/'

# Train const
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 8
__C.TRAIN.EPOCHS = 200*(__C.TRAIN.BATCH_SIZE//8)
__C.TRAIN.D_LR = 5e-5
__C.TRAIN.G_LR = 5e-5
__C.TRAIN.G_TIMING = 4
__C.TRAIN.DECAY_STEP = 10000
__C.TRAIN.PRINT_STEP = 20
__C.TRAIN.CPKT_STEP = 1000
__C.TRAIN.SAVE_VOXEL_STEP = 1000
__C.TRAIN.LOSS = 'BCE'
# __C.TRAIN.LOSS = 'hinge'
__C.TRAIN.LOGDIR = './log/'
__C.TRAIN.ACTIVATION = 'sigmoid'


# Text Const
__C.TEXT = edict()
__C.TEXT.EMBEDDING_DIM = 128

# GAN Const
__C.GAN = edict()
__C.GAN.SN_G = True
__C.GAN.CBN = False
__C.GAN.SEED = 123
__C.GAN.Z_DIM = 8
__C.GAN.NOISE_UNIF_ABS_MAX = 0.5
__C.GAN.EMB_NOISE_DIM = __C.GAN.Z_DIM + __C.TEXT.EMBEDDING_DIM
# if AttnGAN, use 100
# __C.GAN.CONDITION_DIM = 100
__C.GAN.CONDITION_DIM = __C.TEXT.EMBEDDING_DIM
# __C.GAN.CONDITION_DIM = 32
__C.GAN.R_NUM = 2
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
# __C.GAN.GF_DIM = 32
# resplit dataset?
resplit=True

# Stack TREE Configuration
__C.TREE = edict()
__C.TREE.BRANCH_NUM = 2

# https://github.com/kchen92/text2shape/blob/2f62ebc15587f171758c71e3daf0235d7380df41/lib/config.py#L193
def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line).
    """
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d.keys()
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d.keys()
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert (d[subkey] is None) or isinstance(value, type(d[subkey])), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value