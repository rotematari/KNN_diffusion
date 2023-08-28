
import math
from inspect import isfunction
from functools import partial

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize,ToPILImage,Normalize,RandomHorizontalFlip
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from diffusers import UNet2DModel,DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from diffusers import DDPMPipeline


import os
from os.path import isfile, join

from dataclasses import dataclass

from accelerate import Accelerator
from huggingface_hub import HfFolder, Repository, whoami

from tqdm.auto import tqdm
from pathlib import Path
import os
from datetime import datetime
import wandb

from diffusers import UNet2DModel
@dataclass
class TrainingConfig:
    image_size = 32  # the generated image resolution
    train_batch_size = 100
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 1000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 50
    save_model_epochs = 100
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = '/home/robotics20/Documents/deep/KNN_diffusion/unet2dhuggin_face/gen_imgs'  # the model namy locally and on the HF Hub

    # push_to_hub = True  # whether to upload the saved model to the HF Hub
    # hub_private_repo = False
    # overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 42


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid
def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler,load_model_path=None):

    now = datetime.now()
    timestamp = now.strftime('%m_%d_%H_%M')
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        log_with="wandb"
    )
    if accelerator.is_main_process:
    #     if config.push_to_hub:
    #         repo_name = get_full_repo_name(Path(config.output_dir).name)
    #         repo = Repository(config.output_dir, clone_from=repo_name)
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("diffusion",config=config)
        
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if load_model_path is not None :
        
        model = accelerator.unwrap_model(model)
        path_to_checkpoint = os.path.join(load_model_path, "pytorch_model.bin")
        model.load_state_dict(torch.load(path_to_checkpoint))
        

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
        wandb.log(logs)
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch ) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch ) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    pipeline.save_pretrained(config.output_dir)

                    # Save a checkpoint at the end of each epoch
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }

                    path = r'/home/robotics20/Documents/deep/KNN_diffusion/unet2dhuggin_face/results'
                    file_name = f'checkpoint_epoch_{epoch}_time_{timestamp}'
                    accelerator.save_model(model,join(path,file_name))
                    print("saved")



if __name__ == "__main__":

    config = TrainingConfig()
    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
    project="diffusion",
    
    # track hyperparameters and run metadata
    config=config
    )   


    reverse_transform = Compose([
     Lambda(lambda t: t*0.5+0.5),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
    ])

    transforms = Compose([
    
    Resize((config.image_size, config.image_size)),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize([0.5], [0.5]),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root=r"/home/robotics20/Documents/deep/KNN_diffusion/unet2dhuggin_face/rotem/test/data", download=False,
                                            train=True,transform=transforms)

    # create dataloaderrotem/test/data
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.train_batch_size, shuffle=True)

    sample_image = next(iter(train_dataset))
    plt.figure(figsize=(1,1))

    plt.imshow(reverse_transform(sample_image[0]))

    sample_image[0].shape


    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
    )
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    train_loop(config=config,model=model,noise_scheduler=noise_scheduler,optimizer=optimizer,train_dataloader=train_dataloader,lr_scheduler=lr_scheduler,
           load_model_path=r'unet2dhuggin_face/results/checkpoint_epoch_999_time_08_20_20_42.pt')
