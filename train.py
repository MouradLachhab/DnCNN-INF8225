import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import prepare_data, Dataset
from torch.autograd import Variable
import numpy as np
from skimage.measure.simple_metrics import compare_psnr

train_data = Dataset(train=True)

valid_data = Dataset(train=False)

train_loader = DataLoader(dataset=train_data, num_workers=4, batch_size=128, shuffle=True)

def calculate_PSNR(noised_image, clean_image):
    noised_image_cpu = noised_image.data.cpu().numpy().astype(np.float32)
    clean_image_cpu = clean_image.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(noised_image_cpu.shape[0]):
        # function from skimage to calculate PSNR between 2 images
        # data range is use to know the maximum value of the pixel
        PSNR += compare_psnr(clean_image_cpu[i,:,:,:], noised_image_cpu[i,:,:,:], data_range=1.) 
    
    # return mean PSNR between images
    return (PSNR/noised_image_cpu.shape[0])

def train(model, optimizer, criterion):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    loss_training = 0.0
    noise_level=50 # 15,25, 50 sont discuté dans le text. On pourra le rendre optionnel plus tard.

    # Training with SGD ?
    for batch_idx, data in enumerate(train_loader, 0):
        noise = torch.FloatTensor(data.size()).normal_(mean=0, std=noise_level/255.)
        noised_image = data + noise
        # sending values to GPU
        noised_image, data = Variable(noised_image.cuda()), Variable(data.cuda())
        noise = Variable(noise.cuda())

        # Running Model
        output = model(noised_image)

        # TODO: Calculate loss
        loss = criterion(output, noise) / (noised_image.size()[0]*2)
        loss.backward()
        optimizer.step()

        # Le github a model(imgn_train) a la place de output pour X raison. A tester
        # Maybe because they run the backpropagation just before, so they are doing 1 step ahead ?
        # I think in the end there is no difference, we will just be 1 step behind all the time on accuracy
        # This step also seems to be optional as it is only used for results on screen to show progress
        # Calculer l'image denoised
        model.eval()
        denoised_image = torch.clamp(noised_image-output, 0., 1.) 
        psnr = calculate_PSNR(denoised_image, data)
        
        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        loss.backward()
        optimizer.step()

    loss_training /= len(train_data) # averaging the loss for this epoch

    return model, loss_training

def validation(model, criterion):
    model.eval()

    psnr_validation = 0
    loss_validation = 0.0
    noise_level=50 # 15,25, 50 sont discuté dans le text. On pourra le rendre optionnel plus tard.

    for i in range(len(valid_data)): # Placeholder loop
        image = torch.unsqueeze(valid_data[i], 0)
        noise = torch.FloatTensor(image.size()).normal_(mean=0, std=noise_level/255.)
        noised_image = image + noise

        # sending values to GPU
        noised_image, image = Variable(noised_image.cuda()), Variable(image.cuda())

        # Running Model
        output = model(noised_image)

        # Calculer l'image denoised
        denoised_image = torch.clamp(noised_image-output, 0., 1.) 
        psnr_validation += calculate_PSNR(denoised_image, image)

        loss = criterion(output, noise) / (noised_image.size()[0]*2)
        loss_validation += loss.item()

    loss_validation /= len(valid_data)
    psnr_validation /= len(valid_data)

    return psnr_validation, loss_validation


def train_model(model, epochs=50, lr=1e-1): # Need to check how to implement exponential decay
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Mode to GPU
    device_ids = [0]
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    criterion.cuda()

    best_psnr = 0.0
    best_model = None

    for epoch in range(1, epochs + 1):
        model, loss_training = train(model, optimizer, criterion)
        psnr_validation, loss_validation = validation(model, criterion)

        if psnr_validation > best_psnr:
            best_psnr = psnr_validation
            best_model = model

    return best_model, best_psnr
