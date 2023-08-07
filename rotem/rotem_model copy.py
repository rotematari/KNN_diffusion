import torch.nn as nn 
import torch 
import random
import argparse 

parser = argparse.ArgumentParser(description='model Config', add_help=False)

# DN local cnn args 
parser.add_argument("--Dn_in_ch",type=int,default=3)
parser.add_argument("--Dn_midel_ch",type=int,default=64)
parser.add_argument("--Dn_out_ch",type=int,default=8)
parser.add_argument("--Dn_depth",type=int,default=6)
parser.add_argument("--Dn_karnel",type=int,default=3)
parser.add_argument("--Dn_stride",type=int,default=1)

# embading cnn args 
parser.add_argument("--embd_in_ch",type=int,default=8)
parser.add_argument("--embd_midel_ch",type=int,default=64)
parser.add_argument("--embd_out_ch",type=int,default=8)
parser.add_argument("--embd_depth",type=int,default=4)
parser.add_argument("--embd_karnel",type=int,default=3)
parser.add_argument("--embd_stride",type=int,default=1)

# temp cnn args 
parser.add_argument("--temp_in_ch",type=int,default=8)
parser.add_argument("--temp_midel_ch",type=int,default=64)
parser.add_argument("--temp_out_ch",type=int,default=1)
parser.add_argument("--temp_depth",type=int,default=4)
parser.add_argument("--temp_karnel",type=int,default=3)
parser.add_argument("--temp_stride",type=int,default=1)

# D cnn args 
parser.add_argument("--d_patch_size",type=int,default=4)


args_config = parser.parse_args()


random.seed(0)
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

temp_layers = []
dn_layers = []
embd_layers = []
class KNN_model(nn.Module):
    def __init__(self,args_config):

        super(KNN_model, self).__init__()

       ## DN block 
        for i in range(args_config.Dn_depth):

            if i<1:
                in_ch = args_config.Dn_input_ch
                out_ch =args_config.Dn_midel_ch 
    
            elif i == args_config.Dn_depth-1 :

                out_ch = args_config.Dn_out_ch
                 
            else :
                in_ch = args_config.Dn_midel_ch
                out_ch =args_config.Dn_midel_ch 


            dn_layers.extend([
                nn.Conv2d(in_ch,out_ch,3,1,dtype=torch.float32,padding="same"),
                nn.BatchNorm2d(out_ch,dtype=torch.float32),
                nn.ReLU(inplace=True),

            ])
        
        self.dn = nn.Sequential(*dn_layers)   


       
        ## embed block 
        for i in range(args_config.embd_depth):

            if i<1:
                in_ch = args_config.embd_input_ch
                out_ch =args_config.embd_midel_ch 
    
            elif i == args_config.embd_depth-1 :

                out_ch = args_config.embd_out_ch
                 
            else :
                in_ch = args_config.embd_midel_ch
                out_ch =args_config.embd_midel_ch 


            embd_layers.extend([
                nn.Conv2d(in_ch,out_ch,3,1,dtype=torch.float32,padding="same"),
                nn.BatchNorm2d(out_ch,dtype=torch.float32),
                nn.ReLU(inplace=True),

            ])
        
        self.embd = nn.Sequential(*embd_layers)  
       
        ## temp block 
        for i in range(args_config.temp_depth):

            if i<1:
                in_ch = args_config.temp_input_ch
                out_ch =args_config.temp_midel_ch 
    
            elif i == args_config.temp_depth-1 :

                out_ch = args_config.temp_out_ch
                 
            else :
                in_ch = args_config.temp_midel_ch
                out_ch =args_config.temp_midel_ch 


            temp_layers.extend([
                nn.Conv2d(in_ch,out_ch,3,1,dtype=torch.float32,padding="same"),
                nn.BatchNorm2d(out_ch,dtype=torch.float32),
                nn.ReLU(inplace=True),

            ])
        
        self.embd = nn.Sequential(*temp_layers)  

        ### d

        self.unfold = nn.Unfold(kernel_size=( args_config.d_patch_size, args_config.d_patch_size), stride=4 ) 
        
    def make_DorT(self,input_tensor,patch_size = (4, 4),D=True):

        # extract patches using torch.unfold
        patches = input_tensor.unfold(2, patch_size[0], patch_size[0]).unfold(3, patch_size[1], patch_size[1])
        patches = patches.contiguous().view(1, -1, patch_size[0], patch_size[1])

        if D: 
            # compute the Euclidean distance between all pairs of patches
            patch_vectors = patches.view(-1, patch_size[0] * patch_size[1])
            distances = torch.norm(patch_vectors[:, None, :] - patch_vectors[None, :, :], dim=-1)
            return distances

        else:
            # compute the mean of each patch
            means = patches.mean(dim=(2, 3))
            return means
        
        # D = torch.cdist(D,D)
        # D = D.squeeze()

         

    def Continuous_nearest_neighbors_selection(self,D,temp):
    
        r""""
        alpha is 1 row in D*(-1) 
        
        """

        

