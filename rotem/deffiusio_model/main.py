import argparse
import matplotlib.pyplot as plt 

import torch 
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn

import Noise_scheduler
import utils
import model
from model import SimpleUnet


parser = argparse.ArgumentParser(description='Your script description')


# data
parser.add_argument('--data_folder', type=str,default='KNN_diffusion/rotem/deffiusio_model/data/cifar-10-batches-py', help='data file path')
parser.add_argument('--IMG_SIZE', type=int,default=32, help='data file path')
# hyperparameters 
parser.add_argument('--BATCH_SIZE', type=int,default=100 ,help='Output file path')

#deffiusion args 
parser.add_argument('--T', type=int,default=1000 ,help='noising times')
parser.add_argument('--start_betha', type=float, default=0.0001, help='the betha start val')
parser.add_argument('--end_betha', type=float, default=0.02, help='the betha end val')





if __name__ == '__main__':

    config = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = utils.load_transformed_dataset(config=config)
    dataloader = DataLoader(data, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

    utils.Simulate_forward_diffusion(dataloader=dataloader,config=config,num_images=15)

    # plt.show()
    model = SimpleUnet()


    utils.train(config=config,model=model,dataloader=dataloader,device=device)
    plt.show()


