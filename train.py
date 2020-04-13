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

def train(model, X_train, y_train, optimizer, criterion):
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    loss_training = 0.0
    noise_level=50 # 15,25, 50 sont discuté dans le text. On pourra le rendre optionnel plus tard.

    # Training with SGD ?
    for batch_idx, (data, target) in enumerate(train_loader):
        noise = torch.FloatTensor(data.size()).normal_(mean=0, std=noise_level/255.)
        noised_image = data + noise
        # sending values to GPU
        noised_image, data = Variable(noised_image.cuda()), Variable(data.cuda())
        noise = Variable(noise.cuda())

        # Running Model
        output = model(noised_image)

        # TODO: Calculate loss
        #loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
        #loss.backward()
        optimizer.step()
        model.eval()

        # Le github a model(imgn_train) a la place de output pour X raison. A tester
        # Calculer l'image denoised
        denoised_image = torch.clamp(noised_image-output, 0., 1.) 
        psnr = calculate_PSNR(output, data)
        

        optimizer.zero_grad()
        output = model(data)  # calls the forward function
        loss.backward()
        optimizer.step()

    loss_training /= len(X_train) # averaging the loss for this epoch

    return model, loss_training

def validation(model, X_validation, y_validation, criterion):
    model.eval()

    accuracy = 0
    loss_validation = 0.0

    for i in range(0,1): # Placeholder loop  
        # output = model(INPUT)
        # loss = criterion(...)
        loss_validation += loss.item()
        # accuracy +=

    loss_validation /= len(X_validation)
    accuracy /= len(X_validation)

    return accuracy, loss_validation


def train_model(model, epochs=50, lr=1e-1): # Need to check how to implement exponential decay
    optimizer = optim.Adam()
    criterion = nn.MSELoss()

    best_accuracy = 0.0
    best_model = None

    for epoch in range(1, epochs + 1):
        model, loss_training = train(model, X_train, y_train, optimizer, criterion)
        accuracy, loss_validation = validation(model, X_validation, y_validation, criterion)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_model, best_accuracy
