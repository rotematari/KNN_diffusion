import argparse
import matplotlib.pyplot as plt 

import torch 
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn

import Noise_scheduler
import utils
import model
from model import SimpleUnet,build_resunet,Unet




parser = argparse.ArgumentParser(description='Your script description')


# data
parser.add_argument('--data_folder', type=str,default='KNN_diffusion/rotem/deffiusio_model/data/cifar-10-batches-py', help='data file path')
parser.add_argument('--IMG_SIZE', type=int,default=32, help='image size')
parser.add_argument('--channels', type=int,default=3, help='number of channels ')
# hyperparameters 
parser.add_argument('--BATCH_SIZE', type=int,default=124 ,help='batch size')
parser.add_argument('--epochs', type=int,default=500 ,help='epochs')
parser.add_argument('--lr', type=int,default=2e-3 ,help='learning rate ')
parser.add_argument('--weight_decay', type=int,default=0.000 ,help='weight_decay')

#deffiusion args
parser.add_argument('--time_emb_dim', type=int,default=32 ,help='noising times') 
parser.add_argument('--T', type=int,default=1000 ,help='noising times')
parser.add_argument('--start_betha', type=float, default=0.0001, help='the betha start val')
parser.add_argument('--end_betha', type=float, default=0.02, help='the betha end val')





if __name__ == '__main__':

    config = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')

    data = utils.load_transformed_dataset(config=config)

    dataloader = DataLoader(data, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    # plt.ioff()
    # utils.Simulate_forward_diffusion(dataloader=dataloader,config=config,num_images=15)

    # plt.show()
    # model = SimpleUnet(config=config)
    # model = build_resunet(config=config)
    model = Unet(dim= config.IMG_SIZE, channels=config.channels,dim_mults=(1, 2, 4,), use_convnext=False)
    print("Num params: ", sum(p.numel() for p in model.parameters()))
    print(model)
    opt_model = torch.compile(model=model)
    
    utils.train(config=config,model=opt_model,dataloader=dataloader,device=device)


    plt.show()



