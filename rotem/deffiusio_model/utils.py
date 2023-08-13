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
from Noise_scheduler import forward_diffusion_sample,get_index_from_list,init
from datetime import datetime
from os.path import isfile, join


from torch.optim import Adam


def get_loss(model, img, t, device,config,loss_type='l1'):
    x_noisy, noise = forward_diffusion_sample(img, t=t, device=device,config=config)
    noise_pred = model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, noise_pred)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, noise_pred)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, noise_pred)
    else:
        raise NotImplementedError()
    

    return loss

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
                                         train=True, 
                                         transform=data_transform)

    test = torchvision.datasets.CIFAR10(root="/home/robotics20/Documents/deep/KNN_diffusion/rotem/deffiusio_model/data", download=True, 
                                        train=False,
                                         transform=data_transform)
    return torch.utils.data.ConcatDataset([train, test])

@torch.no_grad()
def sample_timestep(x,config, t,model):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    sqrt_alphas_cumprod ,sqrt_one_minus_alphas_cumprod,betas,sqrt_recip_alphas,posterior_variance = init(config=config)

    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 


@torch.no_grad() 
def sample_plot_image(config, device, model):
    # Sample noise
    
    img_size = config.IMG_SIZE # Get the image size from the config object
    T = config.T # Get the number of timesteps from the config object

    # Samples a tensor of size (1, 3, img_size, img_size) with values drawn from a standard normal distribution
    img = torch.randn((1, 3, img_size, img_size), device=device)

    # plt.figure(figsize=(15,15)) 
    # plt.axis('off') 

    num_images = 10 # Set the number of images to display
    stepsize = int(T/num_images) # Calculate the step size for displaying images at regular intervals

    # Iterate over the timesteps in reverse order, from T-1 to 0
    for i in range(0,T)[::-1]:
        # Create a tensor filled with the current timestep value
        t = torch.full((1,), i, device=device, dtype=torch.long) 
        # Update the image by sampling from the current timestep
        img = sample_timestep(x=img, config=config, t=t, model=model) 
        # Clamp the pixel values of the image to be within the range [-1.0, 1.0] to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        # If the current timestep is a multiple of the step size
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu())
    # plt.show() 


# @torch.no_grad()
# def sample_plot_image(config,device,model):
#     # Sample noise
    
#     img_size = config.IMG_SIZE
#     T = config.T
#     #samples an tesnsor in img size as the data in gaussian normal dist 
#     img = torch.randn((1, 3, img_size, img_size), device=device)

#     # plt.figure(figsize=(15,15))
#     plt.axis('off')
#     num_images = 10
#     stepsize = int(T/num_images)
#     # from T to 0 
#     for i in range(0,T)[::-1]:
#         t = torch.full((1,), i, device=device, dtype=torch.long)
#         img = sample_timestep(x=img,config=config, t=t,model=model)
#         # Edit: This is to maintain the natural range of the distribution
#         img = torch.clamp(img, -1.0, 1.0)
#         if i % stepsize == 0:
#             plt.ioff()
#             plt.subplot(1, num_images, int(i/stepsize)+1)
#             show_tensor_image(img.detach().cpu())
            
    
# @torch.compile
def train(config,model,dataloader,device):

    
    plt.ioff()
    plt.figure(figsize=(15,15))
    plt.axis('off')
    # plt.pause(10)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=config.lr,weight_decay=config.weight_decay)
    epochs = config.epochs

    now = datetime.now()
    timestamp = now.strftime('%m_%d_%H_%M')

    for epoch in range(epochs):


        for step, batch in enumerate(dataloader):
            
            batch[0].to(device)

            optimizer.zero_grad()
            # sample a noise timestep 
            t = torch.randint(0, config.T, (config.BATCH_SIZE,), device=device).long()

            loss = get_loss(model, batch[0], t,device=device,config=config,loss_type='huber')
            loss.backward()
            optimizer.step()


            if step%50 == 0 :
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()}")

            
                
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()}")
        sample_plot_image(config=config,device=device,model=model)
        plt.title(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()}")
        plt.pause(0.1)
        # plt.show()

        if epoch%10 ==0:
            # Save a checkpoint at the end of each epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }

            path = r'rotem/deffiusio_model/models '
            file_name = f'checkpoint_epoch_{epoch}_time_{timestamp}.pt'
            torch.save(checkpoint, join(path,file_name)) 

