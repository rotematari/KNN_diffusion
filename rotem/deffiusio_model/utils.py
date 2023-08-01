import matplotlib.pyplot as plt 
import torch 
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms 
import torchvision
import numpy as np
from Noise_scheduler import forward_diffusion_sample


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

def Simulate_forward_diffusion(dataloader,config,num_images=10):


    T = config.T
    # Simulate forward diffusion
    image = next(iter(dataloader))[0]

    plt.figure(figsize=(15,15))
    plt.axis('off')

    stepsize = int(T/num_images)

    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
        img, noise = forward_diffusion_sample(image, t,config)
        show_tensor_image(img)

def load_transformed_dataset(config):
    data_transforms = [
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.CIFAR10(root="/home/robotics20/Documents/deep/KNN_diffusion/rotem/deffiusio_model/data", download=True, 
                                         transform=data_transform)

    test = torchvision.datasets.CIFAR10(root="/home/robotics20/Documents/deep/KNN_diffusion/rotem/deffiusio_model/data", download=True, 
                                         transform=data_transform)
    return torch.utils.data.ConcatDataset([train, test])
