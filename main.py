import os
from dataset import prepare_data, prepare_masks
# Preprocess

otherDataPath = "./DnCNN-PyTorch/data/"

if(not (os.path.exists("./train.h5") and os.path.exists("./val.h5"))):
    train_num_max, val_num_max = prepare_data(otherDataPath, patch_size=40, stride=10, aug_times=1)

if(not (os.path.exists("./train_masks.h5") and os.path.exists("./val_masks.h5"))):
    prepare_masks(patch_size=40, stride=10, train_num_max=train_num_max, val_num_max=val_num_max, aug_times=0)


from train import train_model

# Create model
best_model, best_pnsr = train_model(2, 1e-2, 'noise', False)