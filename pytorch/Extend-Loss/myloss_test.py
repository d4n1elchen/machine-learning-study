import torch
import myloss
from torch.autograd import Variable

loss = myloss.CosineLoss()

input1 = Variable(torch.Tensor([[2, 1], 
                                [3, 4], 
                                [4, 8],
                                [3, 5], 
                                [4, 7]]), requires_grad=True)
input2 = torch.Tensor([[1, 2],
                       [3, 4],
                       [5, 2],
                       [2, 3],
                       [4, 5]])
output = loss(input1, input2)

print(output)

output.backward()

print(input1.grad)