import torch
import torch.nn as nn
import numpy as np
import random as rn
import pickle
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

TESTING_PATH = r'C:\Users\royya\python projects\DeepLearningHW\testfile'

rn.seed(0)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(nn.Module):
    def __init__(self, nearest_neighbors=7, similarity_metrics=3, temporal_depth=1000, image_hight=32, image_width=32, domain_hight=32, domain_width=32, patch_hight=8, patch_width=8, patch_stride=4):
        super(Model, self).__init__()
        self.temporal_depth = temporal_depth
        self.image_hight = image_hight
        self.image_width = image_width
        self.domain_hight = domain_hight
        self.domain_width = domain_width
        self.patch_hight = patch_hight
        self.patch_width = patch_width
        self.patch_stride = patch_stride

        # Folder Unfolder
        self.folder = nn.Fold(output_size=(image_hight, image_width), kernel_size=(
            patch_hight, patch_width), stride=patch_stride)
        self.unfolder = nn.Unfold(kernel_size=(
            patch_hight, patch_width), stride=patch_stride)
        self.fold_unfold_divisor = self.folder(self.unfolder(
            torch.ones(size=(1, 3, image_hight, image_width))))

        # Noising parameters
        
        self.variance_schedual = np.array(
            [(t*(0.02-0.0001)/temporal_depth)+0.0001 for t in range(temporal_depth+1)])
        self.mean_schedual = np.array(
            [np.sqrt(1-var) for var in self.variance_schedual])
        self.alpha_bar = np.array(
            [np.prod(1-self.variance_schedual[:el+1]) for el in range(temporal_depth)])

        # optional leared temporal encoding
        self.temporal_encoding_layer = nn.Linear(
            temporal_depth, 3*image_hight*image_width)

        # first nearest neighbor block

        # first metric
        self.first_metric_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.first_metric_conv_layer_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, padding='valid')
        self.first_metric_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.first_metric_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=40, kernel_size=2, stride=patch_stride, padding='valid')

        # second metric
        self.second_metric_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.second_metric_conv_layer_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, padding='valid')
        self.second_metric_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.second_metric_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=40, kernel_size=2, stride=patch_stride, padding='valid')

        # third metric
        self.third_metric_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.third_metric_conv_layer_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, padding='valid')
        self.third_metric_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.third_metric_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=40, kernel_size=2, stride=patch_stride, padding='valid')

        # temperature
        self.first_temperature_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.first_temperature_conv_layer_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, padding='valid')
        self.first_temperature_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.first_temperature_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=3, kernel_size=2, stride=patch_stride, padding='valid')

        # first sequence of standard layers
        self.first_standard_conv_layer = nn.Conv2d(
            in_channels=3*(similarity_metrics*nearest_neighbors+1), out_channels=64, kernel_size=3, padding='same')
        self.second_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.third_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.fourth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.fifth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.sixth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, padding='same')

        # second nearest neighbor block

        # fourth metric
        self.fourth_metric_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.fourth_metric_conv_layer_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, padding='valid')
        self.fourth_metric_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.fourth_metric_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=40, kernel_size=2, stride=patch_stride, padding='valid')

        # fifth metric
        self.fifth_metric_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.fifth_metric_conv_layer_2 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.fifth_metric_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.fifth_metric_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=40, kernel_size=2, stride=patch_stride, padding='valid')

        # sixth metric
        self.sixth_metric_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.sixth_metric_conv_layer_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, padding='valid')
        self.sixth_metric_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.sixth_metric_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=40, kernel_size=2, stride=patch_stride, padding='valid')

        # second temperature
        self.second_temperature_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.second_temperature_conv_layer_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, padding='valid')
        self.second_temperature_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.second_temperature_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=3, kernel_size=2, stride=patch_stride, padding='valid')

        # second sequence of standard layers
        self.seventh_standard_conv_layer = nn.Conv2d(
            in_channels=3*(similarity_metrics*nearest_neighbors+1), out_channels=64, kernel_size=3, padding='same')
        self.eigth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.ninth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.tenth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.eleventh_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.twelveth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, padding='same')

        # third nearest neighbor block

        # seventh metric
        self.seventh_metric_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.seventh_metric_conv_layer_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, padding='valid')
        self.seventh_metric_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.seventh_metric_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=40, kernel_size=2, stride=patch_stride, padding='valid')

        # eigth metric
        self.eigth_metric_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.eigth_metric_conv_layer_2 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.eigth_metric_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.eigth_metric_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=40, kernel_size=2, stride=patch_stride, padding='valid')

        # ninth metric
        self.ninth_metric_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.ninth_metric_conv_layer_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, padding='valid')
        self.ninth_metric_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.ninth_metric_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=40, kernel_size=2, stride=patch_stride, padding='valid')

        # third temperature
        self.third_temperature_conv_layer_1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=3, padding='valid')
        self.third_temperature_conv_layer_2 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=3, padding='valid')
        self.third_temperature_conv_layer_3 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=3, padding='valid')
        self.third_temperature_conv_layer_4 = nn.Conv2d(
            in_channels=30, out_channels=3, kernel_size=2, stride=patch_stride, padding='valid')

        # third sequence of standard layers
        self.thirteenth_standard_conv_layer = nn.Conv2d(
            in_channels=3*(similarity_metrics*nearest_neighbors+1), out_channels=64, kernel_size=3, padding='same')
        self.fourteenth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.fifteenth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.sixteenth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.seventeenth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding='same')
        self.eighteenth_standard_conv_layer = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=3, padding='same')

        # batch normalisations
        self.first_batch_norm = nn.BatchNorm2d(num_features=64, affine=True)
        self.second_batch_norm = nn.BatchNorm2d(num_features=64, affine=True)
        self.third_batch_norm = nn.BatchNorm2d(num_features=64, affine=True)
        self.fourth_batch_norm = nn.BatchNorm2d(num_features=64, affine=True)
        self.fifth_batch_norm = nn.BatchNorm2d(num_features=64, affine=True)
        self.sixth_batch_norm = nn.BatchNorm2d(num_features=3, affine=True)

        self.seventh_batch_norm = nn.BatchNorm2d(num_features=64, affine=True)
        self.eigth_batch_norm = nn.BatchNorm2d(num_features=64, affine=True)
        self.ninth_batch_norm = nn.BatchNorm2d(num_features=64, affine=True)
        self.tenth_batch_norm = nn.BatchNorm2d(num_features=64, affine=True)
        self.eleventh_batch_norm = nn.BatchNorm2d(num_features=64, affine=True)
        self.twelveth_batch_norm = nn.BatchNorm2d(num_features=3, affine=True)

        self.thirteenth_batch_norm = nn.BatchNorm2d(
            num_features=64, affine=True)
        self.fourteenth_batch_norm = nn.BatchNorm2d(
            num_features=64, affine=True)
        self.fifteenth_batch_norm = nn.BatchNorm2d(
            num_features=64, affine=True)
        self.sixteenth_batch_norm = nn.BatchNorm2d(
            num_features=64, affine=True)
        self.seventeenth_batch_norm = nn.BatchNorm2d(
            num_features=64, affine=True)
        self.eighteenth_batch_norm = nn.BatchNorm2d(
            num_features=3, affine=True)

        # activation function
        self.relu = nn.ReLU()

        # loss functions
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.huber_loss = nn.HuberLoss()

    # TEMPORAL ENCODING METHODS:

    def static_temporal_encoding(self, input_tensor, time_list):
        B, _, _, _ = input_tensor.size()
        hight_temporal_encoding = []
        width_temporal_encoding = []

        for time in time_list:
            hight_encoding = []
            for i in range(self.image_hight):
                if i % 2 == 0:
                    hight_encoding.append(np.sin(time/(i/self.image_hight)))
                else:
                    hight_encoding.append(np.cos(time/((i-1)/self.image_hight)))
            width_encoding = []
            for j in range(self.image_width):
                if j % 2 == 0:
                    width_encoding.append(np.sin(time/(j/self.image_hight)))
                else:
                    width_encoding.append(
                        np.cos(time/((j-1)/self.image_hight)))
            hight_temporal_encoding.append(hight_encoding)
            width_temporal_encoding.append(width_encoding)

        hight_temporal_encoding = torch.tensor(
            hight_temporal_encoding, device=device, dtype=torch.float32)
        width_temporal_encoding = torch.tensor(
            width_temporal_encoding, device=device, dtype=torch.float32)

        hight_temporal_encoding = hight_temporal_encoding.view(
            B, 1, self.image_hight, 1)
        width_temporal_encoding = width_temporal_encoding.view(
            B, 1, 1, self.image_width)

        total_temporal_encoding = hight_temporal_encoding*width_temporal_encoding

        output = input+total_temporal_encoding
        return output

    def dynamic_temporal_encoding(self, time_hot_one_encoding):
        B, _ = time_hot_one_encoding.size()
        out = self.temporal_encoding_layer(time_hot_one_encoding)
        out = out.view(B, self.image_hight, self.image_width)
        return out

    # FORWARD PASS METHODS:

    def nearest_neighbor_forward(self, input, temperature_layers, first_metric_layers, second_metric_layers, third_metric_layers):
        # assining constants used for numerical stability
        log_epsilon = torch.tensor(0.1**45, dtype=torch.float32)
        dev_epsilon = torch.tensor(0.01, dtype=torch.float32)

        # temperatures processing
        temperatures = temperature_layers[0](input)
        temperatures = temperature_layers[1](temperatures)
        temperatures = temperature_layers[2](temperatures)
        temperatures = torch.abs(temperature_layers[3](temperatures))

        B, C, H, W = temperatures.size()
        temperatures = temperatures.view(B, C, H, W, 1, 1)

        # first metric processing
        metric_1 = first_metric_layers[0](input)
        metric_1 = first_metric_layers[1](metric_1)
        metric_1 = first_metric_layers[2](metric_1)
        metric_1 = first_metric_layers[3](metric_1)

        B, C, H, W = metric_1.size()
        distance_tensor = torch.pow(metric_1.view(
            B, C, H, W, 1, 1)-metric_1.view(B, C, 1, 1, H, W), 2)
        distance_tensor = torch.sum(distance_tensor, dim=1)
        # change dimension for normalisation purpuses and for summing over the different patches at the end
        distance_tensor = distance_tensor.view(B, H, W, H*W, 1)

        # perform nearest neighbor distribution calculation
        first_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(dev_epsilon+temperatures[:, 0, :, :, :, :])), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     first_nearest_neighbor_tensor+log_epsilon)
        second_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(dev_epsilon+temperatures[:, 0, :, :, :, :])), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     second_nearest_neighbor_tensor+log_epsilon)
        third_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(dev_epsilon+temperatures[:, 0, :, :, :, :])), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     third_nearest_neighbor_tensor+log_epsilon)
        fourth_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(dev_epsilon+temperatures[:, 0, :, :, :, :])), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     fourth_nearest_neighbor_tensor+log_epsilon)
        fifth_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(dev_epsilon+temperatures[:, 0, :, :, :, :])), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     fifth_nearest_neighbor_tensor+log_epsilon)
        sixth_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(dev_epsilon+temperatures[:, 0, :, :, :, :])), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     sixth_nearest_neighbor_tensor+log_epsilon)
        seventh_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(dev_epsilon+temperatures[:, 0, :, :, :, :])), p=1, dim=3)
        # set up folder and unfolder
        Unfolder = self.unfolder
        Folder = self.folder
        divisor = self.fold_unfold_divisor

        # performing nearest neighbor averaging
        unfolded_input = Unfolder(input).view(
            B, 1, 1, H*W, 3*self.patch_hight*self.patch_width)
        unfolded_input = torch.broadcast_to(
            unfolded_input, (B, H, W, H*W, 3*self.patch_hight*self.patch_width))

        first_nearest_neighbor_tensor = torch.sum(
            first_nearest_neighbor_tensor*unfolded_input, dim=3)
        first_nearest_neighbor_tensor = Folder(
            first_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        second_nearest_neighbor_tensor = torch.sum(
            second_nearest_neighbor_tensor*unfolded_input, dim=3)
        second_nearest_neighbor_tensor = Folder(
            second_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        third_nearest_neighbor_tensor = torch.sum(
            third_nearest_neighbor_tensor*unfolded_input, dim=3)
        third_nearest_neighbor_tensor = Folder(
            third_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        fourth_nearest_neighbor_tensor = torch.sum(
            fourth_nearest_neighbor_tensor*unfolded_input, dim=3)
        fourth_nearest_neighbor_tensor = Folder(
            fourth_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        fifth_nearest_neighbor_tensor = torch.sum(
            fifth_nearest_neighbor_tensor*unfolded_input, dim=3)
        fifth_nearest_neighbor_tensor = Folder(
            fifth_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        sixth_nearest_neighbor_tensor = torch.sum(
            sixth_nearest_neighbor_tensor*unfolded_input, dim=3)
        sixth_nearest_neighbor_tensor = Folder(
            sixth_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        seventh_nearest_neighbor_tensor = torch.sum(
            seventh_nearest_neighbor_tensor*unfolded_input, dim=3)
        seventh_nearest_neighbor_tensor = Folder(
            seventh_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        output = torch.cat((input, first_nearest_neighbor_tensor, second_nearest_neighbor_tensor, third_nearest_neighbor_tensor,
                           fourth_nearest_neighbor_tensor, fifth_nearest_neighbor_tensor, sixth_nearest_neighbor_tensor, seventh_nearest_neighbor_tensor), dim=1)

    # second metric processing
        metric_2 = second_metric_layers[0](input)
        metric_2 = second_metric_layers[1](metric_2)
        metric_2 = second_metric_layers[2](metric_2)
        metric_2 = second_metric_layers[3](metric_2)

        B, C, H, W = metric_2.size()
        distance_tensor = torch.pow(metric_2.view(
            B, C, H, W, 1, 1)-metric_2.view(B, C, 1, 1, H, W), 2)
        distance_tensor = torch.sum(distance_tensor, dim=1)
        # change dimension for normalisation purpuses and for summing over the different patches at the end
        distance_tensor = distance_tensor.view(B, H, W, H*W, 1)

        # perform nearest neighbor distribution calculation
        first_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 1, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     first_nearest_neighbor_tensor+log_epsilon)
        second_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 1, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     second_nearest_neighbor_tensor+log_epsilon)
        third_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 1, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     third_nearest_neighbor_tensor+log_epsilon)
        fourth_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 1, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     fourth_nearest_neighbor_tensor+log_epsilon)
        fifth_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 1, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     fifth_nearest_neighbor_tensor+log_epsilon)
        sixth_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 1, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     sixth_nearest_neighbor_tensor+log_epsilon)
        seventh_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 1, :, :, :, :]+dev_epsilon)), p=1, dim=3)

        # performing nearest neighbor averaging

        unfolded_input = Unfolder(input).view(
            B, 1, 1, H*W, 3*self.patch_hight*self.patch_width)
        unfolded_input = torch.broadcast_to(
            unfolded_input, (B, H, W, H*W, 3*self.patch_hight*self.patch_width))

        first_nearest_neighbor_tensor = torch.sum(
            first_nearest_neighbor_tensor*unfolded_input, dim=3)
        first_nearest_neighbor_tensor = Folder(
            first_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        second_nearest_neighbor_tensor = torch.sum(
            second_nearest_neighbor_tensor*unfolded_input, dim=3)
        second_nearest_neighbor_tensor = Folder(
            second_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        third_nearest_neighbor_tensor = torch.sum(
            third_nearest_neighbor_tensor*unfolded_input, dim=3)
        third_nearest_neighbor_tensor = Folder(
            third_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        fourth_nearest_neighbor_tensor = torch.sum(
            fourth_nearest_neighbor_tensor*unfolded_input, dim=3)
        fourth_nearest_neighbor_tensor = Folder(
            fourth_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        fifth_nearest_neighbor_tensor = torch.sum(
            fifth_nearest_neighbor_tensor*unfolded_input, dim=3)
        fifth_nearest_neighbor_tensor = Folder(
            fifth_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        sixth_nearest_neighbor_tensor = torch.sum(
            sixth_nearest_neighbor_tensor*unfolded_input, dim=3)
        sixth_nearest_neighbor_tensor = Folder(
            sixth_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        seventh_nearest_neighbor_tensor = torch.sum(
            seventh_nearest_neighbor_tensor*unfolded_input, dim=3)
        seventh_nearest_neighbor_tensor = Folder(
            seventh_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        output = torch.cat((output, first_nearest_neighbor_tensor, second_nearest_neighbor_tensor, third_nearest_neighbor_tensor,
                           fourth_nearest_neighbor_tensor, fifth_nearest_neighbor_tensor, sixth_nearest_neighbor_tensor, seventh_nearest_neighbor_tensor), dim=1)

    # third metric processing
        metric_3 = third_metric_layers[0](input)
        metric_3 = third_metric_layers[1](metric_3)
        metric_3 = third_metric_layers[2](metric_3)
        metric_3 = third_metric_layers[3](metric_3)

        B, C, H, W = metric_3.size()
        distance_tensor = torch.pow(metric_3.view(
            B, C, H, W, 1, 1)-metric_3.view(B, C, 1, 1, H, W), 2)
        distance_tensor = torch.sum(distance_tensor, dim=1)
        # change dimension for normalisation purpuses and for summing over the different patches at the end
        distance_tensor = distance_tensor.view(B, H, W, H*W, 1)

        # perform nearest neighbor distribution calculation
        first_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 2, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     first_nearest_neighbor_tensor+log_epsilon)
        second_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 2, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     second_nearest_neighbor_tensor+log_epsilon)
        third_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 2, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     third_nearest_neighbor_tensor+log_epsilon)
        fourth_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 2, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     fourth_nearest_neighbor_tensor+log_epsilon)
        fifth_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 2, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     fifth_nearest_neighbor_tensor+log_epsilon)
        sixth_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 2, :, :, :, :]+dev_epsilon)), p=1, dim=3)
        distance_tensor += torch.log(1 -
                                     sixth_nearest_neighbor_tensor+log_epsilon)
        seventh_nearest_neighbor_tensor = nn.functional.normalize(
            torch.exp(distance_tensor/(temperatures[:, 2, :, :, :, :]+dev_epsilon)), p=1, dim=3)

        # performing nearest neighbor averaging

        unfolded_input = Unfolder(input).view(
            B, 1, 1, H*W, 3*self.patch_hight*self.patch_width)
        unfolded_input = torch.broadcast_to(
            unfolded_input, (B, H, W, H*W, 3*self.patch_hight*self.patch_width))

        first_nearest_neighbor_tensor = torch.sum(
            first_nearest_neighbor_tensor*unfolded_input, dim=3)
        first_nearest_neighbor_tensor = Folder(
            first_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        second_nearest_neighbor_tensor = torch.sum(
            second_nearest_neighbor_tensor*unfolded_input, dim=3)
        second_nearest_neighbor_tensor = Folder(
            second_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        third_nearest_neighbor_tensor = torch.sum(
            third_nearest_neighbor_tensor*unfolded_input, dim=3)
        third_nearest_neighbor_tensor = Folder(
            third_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        fourth_nearest_neighbor_tensor = torch.sum(
            fourth_nearest_neighbor_tensor*unfolded_input, dim=3)
        fourth_nearest_neighbor_tensor = Folder(
            fourth_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        fifth_nearest_neighbor_tensor = torch.sum(
            fifth_nearest_neighbor_tensor*unfolded_input, dim=3)
        fifth_nearest_neighbor_tensor = Folder(
            fifth_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        sixth_nearest_neighbor_tensor = torch.sum(
            sixth_nearest_neighbor_tensor*unfolded_input, dim=3)
        sixth_nearest_neighbor_tensor = Folder(
            sixth_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        seventh_nearest_neighbor_tensor = torch.sum(
            seventh_nearest_neighbor_tensor*unfolded_input, dim=3)
        seventh_nearest_neighbor_tensor = Folder(
            seventh_nearest_neighbor_tensor.view(B, 3*self.patch_hight*self.patch_width, H*W))/divisor

        output = torch.cat((output, first_nearest_neighbor_tensor, second_nearest_neighbor_tensor, third_nearest_neighbor_tensor,
                           fourth_nearest_neighbor_tensor, fifth_nearest_neighbor_tensor, sixth_nearest_neighbor_tensor, seventh_nearest_neighbor_tensor), dim=1)
        return output

    def standart_layers_forward(self, standart_layers, batchnorm_layers, input):
        temp_out = standart_layers[0](input)
        temp_out = self.relu(temp_out)
        temp_out = batchnorm_layers[0](temp_out)
        out = temp_out+input
        temp_out = standart_layers[1](out)
        temp_out = self.relu(temp_out)
        temp_out = batchnorm_layers[1](temp_out)
        out += temp_out
        temp_out = standart_layers[2](out)
        temp_out = self.relu(temp_out)
        temp_out = batchnorm_layers[2](temp_out)
        out += temp_out
        temp_out = standart_layers[3](out)
        temp_out = self.relu(temp_out)
        temp_out = batchnorm_layers[3](temp_out)
        out += temp_out
        temp_out = standart_layers[4](out)
        temp_out = self.relu(temp_out)
        temp_out = batchnorm_layers[4](temp_out)
        out += temp_out
        temp_out = standart_layers[5](out)
        temp_out = self.relu(temp_out)
        temp_out = batchnorm_layers[5](temp_out)
        out += temp_out
        return out

    def estimate_epsilon(self, input, t, static_temporal_encoding=False):
        if static_temporal_encoding:
            temporal_encoding = self.static_temporal_encoding(t)
        else:
            temporal_encoding = self.dynamic_temporal_encoding(
                t)+self.static_temporal_encoding(nn.functional.one_hot(torch.tensor(t, dtype=torch.long, device=device), num_classes=self.temporal_depth))

        out = self.nearest_neighbor_forward(input, [self.first_temperature_conv_layer_1, self.first_temperature_conv_layer_2, self.first_temperature_conv_layer_3, self.first_temperature_conv_layer_4], [self.first_metric_conv_layer_1, self.first_metric_conv_layer_2, self.first_metric_conv_layer_3, self.first_metric_conv_layer_4], [
                                            self.second_metric_conv_layer_1, self.second_metric_conv_layer_2, self.second_metric_conv_layer_3, self.second_metric_conv_layer_4], [self.third_metric_conv_layer_1, self.third_metric_conv_layer_2, self.third_metric_conv_layer_3, self.third_metric_conv_layer_4])

        out = self.standart_layers_forward([self.first_standard_conv_layer, self.second_standard_conv_layer, self.third_standard_conv_layer, self.fourth_standard_conv_layer, self.fifth_standard_conv_layer, self.sixth_standard_conv_layer], [
                                           self.first_batch_norm, self.second_batch_norm, self.third_batch_norm, self.fourth_batch_norm, self.fifth_batch_norm, self.sixth_batch_norm], out+temporal_encoding)

        out = self.nearest_neighbor_forward(out, [self.second_temperature_conv_layer_1, self.second_temperature_conv_layer_2, self.second_temperature_conv_layer_3, self.second_temperature_conv_layer_4], [self.fourth_metric_conv_layer_1, self.fourth_metric_conv_layer_2, self.fourth_metric_conv_layer_3, self.fourth_metric_conv_layer_4], [
                                            self.fifth_metric_conv_layer_1, self.fifth_metric_conv_layer_2, self.fifth_metric_conv_layer_3, self.fifth_metric_conv_layer_4], [self.sixth_metric_conv_layer_1, self.sixth_metric_conv_layer_2, self.sixth_metric_conv_layer_3, self.sixth_metric_conv_layer_4])

        out = self.standart_layers_forward([self.seventh_standard_conv_layer, self.eigth_standard_conv_layer, self.ninth_standard_conv_layer, self.tenth_standard_conv_layer, self.eleventh_standard_conv_layer, self.twelveth_standard_conv_layer], [
                                           self.seventh_batch_norm, self.eigth_batch_norm, self.ninth_batch_norm, self.tenth_batch_norm, self.eleventh_batch_norm, self.twelveth_batch_norm], out+temporal_encoding)

        out = self.nearest_neighbor_forward(out, [self.third_temperature_conv_layer_1, self.third_temperature_conv_layer_2, self.third_temperature_conv_layer_3, self.third_temperature_conv_layer_4], [self.seventh_metric_conv_layer_1, self.seventh_metric_conv_layer_2, self.seventh_metric_conv_layer_3, self.seventh_metric_conv_layer_4], [
                                            self.eigth_metric_conv_layer_1, self.eigth_metric_conv_layer_2, self.eigth_metric_conv_layer_3, self.eigth_metric_conv_layer_4], [self.ninth_metric_conv_layer_1, self.ninth_metric_conv_layer_2, self.ninth_metric_conv_layer_3, self.ninth_metric_conv_layer_4])

        out = self.standart_layers_forward([self.thirteenth_standard_conv_layer, self.fourteenth_standard_conv_layer, self.fifteenth_standard_conv_layer, self.sixteenth_standard_conv_layer, self.seventeenth_standard_conv_layer, self.eighteenth_standard_conv_layer], [
                                           self.thirteenth_batch_norm, self.fourteenth_batch_norm, self.fifteenth_batch_norm, self.sixteenth_batch_norm, self.seventeenth_batch_norm, self.eighteenth_batch_norm], out+temporal_encoding)
        return out

    # IMAGE SAMPLING METHODS:

    def sample_noised_image(self, input_image, t):
        output = torch.normal(
            mean=self.alpha_bar[t]*input_image, std=torch.sqrt(1-self.alpha_bar[t]))
        return output

    def sample_epsilon(self, input_image, t):
        noised_imaged = self.sample_noised_image(input_image, t)
        epsilon = (
            noised_imaged-np.sqrt(self.alpha_bar[t])*input_image/np.sqrt(1-self.alpha_bar[t]))
        return epsilon

    def sample_one_step_denoising(self, input, estimated_epsilon, t):
        normaly_distributed_tensor = torch.normal(mean=torch.zeros(
            (self.image_hight, self.image_width)), std=np.sqrt(self.variance_schedual[t]))
        output = (1/np.sqrt(self.variance_schedual[t]))*(input-(self.variance_schedual[t]/np.sqrt(
            1-self.alpha_bar[t]))*estimated_epsilon)+normaly_distributed_tensor
        return output

    def denoise_image(self, input, t, static_temporal_encoding=False):
        if t == 0:
            return input
        else:
            new_image = self.sample_one_step_denoising(
                input, self.estimate_epsilon(input, t, static_temporal_encoding=static_temporal_encoding), t)
            return self.denoise_image(new_image, t-1)
