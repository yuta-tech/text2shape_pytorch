# text2shape_pytorch  
## Unofficial Pytorch Implementation of Text2Shape  
This repository has only voxel generation part.  
voxel data : http://text2shape.stanford.edu/dataset/shapenet/nrrd_256_filter_div_32_solid.zip  
caption pickles : http://text2shape.stanford.edu/dataset/text2shape-data.zip  
change DIR.DATA_PATH, DIR.RGB_VOXEL_PATH in utils/config.py  
  
### Training
train : sh scripts/train_t2s.sh  
summary model : sh scripts/model_summary.sh  
  
### Notes  
This is unofficial repository, so No evaluate code is available.  
It means this repository doesn't reproduce completly.