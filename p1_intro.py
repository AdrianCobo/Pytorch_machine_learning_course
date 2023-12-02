import torch

x = torch.Tensor([5,3])
y = torch.Tensor([2,1])

# result [5*2,3*1]
print(x*y)

# matrix 2x5 size with all 0s
x = torch.zeros([2,5])
print(x, x.shape)

# generate 2x5 random matrix
y = torch.rand([2,5])
print(y)

# resape matrix with view from 2x5 to 10x1
y = y.view([1,10])