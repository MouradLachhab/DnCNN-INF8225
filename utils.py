import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

save_dir = "./models/"
model_name = save_dir + "best_model.pth"

def weights_init_kaiming(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != - 1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != - 1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

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

def add_noise(data):
    noise = torch.FloatTensor(data.size()).normal_(mean=0, std=noise_level/255.)
    noised_image = data + noise
    noise = Variable(noise.cuda())

    return noised_image, noise

def add_mask(data, masks):
    masked_image = torch.clamp(data + masks, 0., 1.)

    return masked_image, data

def show_tensor(image_name, data):
    return

def findLastPSNR(save_directory):
    file_list = glob.glob(os.path.join(save_dir, 'best_model_*.pth'))
    if file_list:
        psnr_list = []
        for file_ in file_list:
            result = re.findall("best_model_(.*).pth.*", file_)
            psnr_list.append(int(result[0]))
        best_psnr = max(psnr_list)
    else:
        best_psnr = 0
    return float(best_psnr)

def findLastLoss(save_directory):
    file_list = glob.glob(os.path.join(save_dir, 'best_model_*.pth'))
    if file_list:
        loss_list = []
        for file_ in file_list:
            result = re.findall("best_model_(.*).pth.*", file_)
            loss_list.append(int(result[0]))
        best_loss = min(loss_list)
    else:
        best_loss = 0
    return float(best_loss)