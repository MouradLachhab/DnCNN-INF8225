import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import DnCNN
from torch.utils.data import DataLoader
from dataset import prepare_data, Dataset
from torch.autograd import Variable
from skimage.measure.simple_metrics import compare_psnr
from utils import add_mask, add_noise, calculate_PSNR, findLastLoss, findLastPSNR, show_tensor, weights_init_kaiming
from datetime import datetime

save_dir = "./models/"
model_name = save_dir + "best_model.pth"

train_data = Dataset(train=True)
valid_data = Dataset(train=False)
train_loader = DataLoader(dataset=train_data, num_workers=4, batch_size=128, shuffle=True)

train_data_masks = Dataset(train=True, mask=True)
valid_data_masks = Dataset(train=False, mask=True)
mask_loader = DataLoader(dataset=train_data_masks, num_workers=4, batch_size=128, shuffle=True)
valid__mask_loader = DataLoader(dataset=valid_data_masks, num_workers=1, batch_size=12, shuffle=True)

def train(model, optimizer, criterion, purpose):
    print("Train Current Time =", datetime.now().strftime("%H:%M:%S"))
    model.train()

    loss_training = 0.0
    mask_loader_2 = iter(mask_loader) 
    for batch_idx, data in enumerate(train_loader, 0):
        optimizer.zero_grad()
        model.zero_grad()

        if purpose == 'noise':
            altered_image, noise = add_noise(data)
        else:
            masks = next(mask_loader_2)
            altered_image, data = add_mask(data, masks)

        altered_image, data = Variable(altered_image.cuda()), Variable(data.cuda())

        output = model(altered_image)

        if purpose == 'noise':
            loss = criterion(output, noise) / (altered_image.size()[0]*2)
            reconstructed_image = torch.clamp(altered_image-output, 0., 1.) 
            psnr = calculate_PSNR(reconstructed_image, data)
        else:
            loss = criterion(output, data)

        loss.backward()
        optimizer.step()
        break
    return model, loss_training

def validation(model, criterion, purpose):
    print("Validate")
    model.eval()

    psnr_validation = 0
    loss_validation = 0.0

    valid__mask_loader_2 = iter(valid__mask_loader) 

    for batch_idx, data in enumerate(valid_loader, 0):
        print(data.size())
        if purpose == 'noise':
            image = torch.unsqueeze(image, 0)
            altered_image, noise = add_noise(image)

        else:
            masks = next(valid__mask_loader_2)
            altered_image, image = add_mask(data, masks)
        
        altered_image, image = Variable(altered_image.cuda()), Variable(image.cuda())

        output = model(altered_image)

        if purpose == 'noise':
            loss = criterion(output, noise) / (altered_image.size()[0]*2)
            reconstructed_image = torch.clamp(altered_image-output, 0., 1.) 
            psnr_validation += calculate_PSNR(reconstructed_image, image)
        else:
            loss = criterion(output, image)

        for i in range(data.size()[0]):
            show_tensor("Original", data[i])

            if purpose == 'noise':
                show_tensor("Noise", noise[i])
                show_tensor("Predicted Noise", output[i])
                show_tensor("Noisy Image", altered_image[i])
                show_tensor("Denoised", reconstructed_image[i])
            else:
                show_tensor("Mask", masks[i])
                show_tensor("Masked image", altered_image[i])
                show_tensor("Filled", output[i])

        loss_validation += loss.item()

    return psnr_validation, loss_validation


def train_model(epochs=50, lr=1e-2, purpose='noise', load = False): 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if load:
        print("loading model")
        if purpose == "noise":
            best_psnr = findLastPSNR(save_dir)
            model = torch.load(save_dir + "best_model_%03d.pth" %(int(best_psnr)))
            best_model = model
        elif purpose == "filling":
            best_loss = findLastLoss(save_dir)/1000
            model = torch.load(save_dir + "best_model_%03d.pth" %(int(best_loss)))
            best_model = model
        print("model loaded")

    elif purpose == 'noise' or purpose == 'filling':
        model = DnCNN(channels=1)
        model.apply(weights_init_kaiming)
        best_psnr = 0.0
        best_loss = 9999999
        best_model = None
    else:
        print("Invalid model")
        return

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    device_ids = [0]
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    criterion.cuda()

    for epoch in range(1, epochs + 1):
        model, loss_training = train(model, optimizer, criterion, purpose)
        psnr_validation, loss_validation = validation(model, criterion, purpose)
        if epoch < 30:
            current_lr = lr
        else:
            current_lr = lr / 2.

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        print("[epoch %d]loss: %.4f PSNR_validation: %.4f" % (epoch, loss_validation, psnr_validation))

        if (psnr_validation > best_psnr and purpose=='noise') or (purpose=='filling' and loss_validation < best_loss):
            best_psnr = psnr_validation
            best_model = model
            best_loss = loss_validation
            print("saving model")
            suffix = 0
            if purpose == "noise":
                suffix = int(best_psnr)
            elif purpose == "filling":
                suffix = int(best_loss * 1000)

            model_name = os.path.join(save_dir, "best_model_%03d.pth" % (suffix))
            if os.path.exists(model_name):
                os.remove(model_name)
            torch.save(model, model_name)     
    return best_model, best_psnr