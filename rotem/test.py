import matplotlib.pyplot as plt 
import torch 

print(torch.__version__)

@torch.compile(fullgraph=True)
def foo(x):
    return torch.sin(x) + torch.cos(x)

foo(torch.tensor(5))
# list = [1,2,3,4,5]

# plt.plot(list)

# for i in range(100000):

#     list.append(i)
#     plt.plot(list)
#     plt.pause(0.0001)