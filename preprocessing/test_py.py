import torch

a = torch.Tensor([75570.9609, 66423.6094,  1096.9343,  1067.2501,   568.6036,   566.6863,
          741.5537,  1212.5231,  1024.2263,   239.9711,  5421.3887,   792.8462,
         1084.5452,  1782.6764, 10304.4121,    88.0704,  8401.6338, 83404.1250,
         2351.9792])
a_max = a.max()
a_min = a.min()
b = a_max/a
print(b)