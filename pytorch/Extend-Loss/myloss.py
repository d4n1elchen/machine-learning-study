import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module

class CosineLoss(Module):

    def __init__(self, reduction='mean'):
        super(CosineLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        output = F.cosine_similarity(input, target)
        
        if self.reduction == 'mean':
            output_reduced = output.mean()
        else:
            raise RuntimeError(f"Reduction: {self.reduction} does not support.")
        return output_reduced