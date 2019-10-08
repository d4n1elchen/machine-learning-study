import torch
import torch.nn.functional as F
from torch.autograd import Variable

input1 = Variable(torch.Tensor([[1, 2], 
                                [3, 4], 
                                [5, 2],
                                [2, 3], 
                                [4, 5]]), requires_grad=True)
input2 = torch.Tensor([[1, 2],
                       [3, 4],
                       [5, 2],
                       [2, 3],
                       [4, 5]])

output = F.cosine_similarity(input1, input2)
output_reduced = output.mean()

print(output)
print(output_reduced)

output_reduced.backward()

print(input1.grad)