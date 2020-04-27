import torch.utils.data as udata
import h5py
import cv2
import os
import glob
import numpy as np
import random

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))
    
def prepare_data(data_path, patch_size, stride, aug_times=1):
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    testPath = os.path.join(data_path, 'train', '*.png')
    files = glob.glob(testPath)
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            #print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        #print("file: %s" % files[i])
        img = cv2.imread(files[i])
        if img.shape[1] == 512:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)

        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

    return train_num, val_num

def prepare_masks(patch_size, stride, train_num_max, val_num_max, aug_times=1):
    print('process training data')
    scales = [0.3]
    testPath = os.path.join('irregular_mask', 'disocclusion_img_mask', '*.png')
    files = glob.glob(testPath)
    files.sort()
    h5f = h5py.File('train_masks.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img = np.float32(normalize(Img))
            ones = np.ones(Img.shape)
            Img = ones - Img
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            #print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
                    if train_num == train_num_max:
                        break
                if train_num == train_num_max:
                   break
            if train_num == train_num_max:
                break
        if train_num == train_num_max:
            print("capped train masks")
            break

    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join('mask', 'testing_mask_dataset', '*.png'))
    files.sort()
    h5f = h5py.File('val_masks.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        #print("file: %s" % files[i])
        img = cv2.imread(files[i])
        h, w, c = img.shape
        img = img[0:256,0:256]
        img = np.float32(normalize(img))
        img = np.expand_dims(img[:,:,0], 0)
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
        if val_num == val_num_max:
          print("capped val masks")
          break
    h5f.close()
    print('training mask set, # samples %d\n' % train_num)
    print('val mask set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True, mask=False):
        super(Dataset, self).__init__()
        self.train = train
        self.mask = mask
        if self.train and not self.mask:
            h5f = h5py.File('train.h5', 'r')
        elif self.train and self.mask:
            h5f = h5py.File('train_masks.h5', 'r')
        elif not self.train and not self.mask:
            h5f = h5py.File('val.h5', 'r')
        elif not self.train and self.mask:
            h5f = h5py.File('val_masks.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train and not self.mask:
            h5f = h5py.File('train.h5', 'r')
        elif self.train and self.mask:
            h5f = h5py.File('train_masks.h5', 'r')
        elif not self.train and not self.mask:
            h5f = h5py.File('val.h5', 'r')
        elif not self.train and self.mask:
            h5f = h5py.File('val_masks.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)