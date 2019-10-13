import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Module

class CosineLoss(Module):
    r"""Returns cosine loss between :math:`input` and :math:`target`, computed along dim.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.

    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same shape as the Input1
        - Output: scalar
    Examples::
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> loss = myloss.CosineLoss(dim=1, eps=1e-6)
        >>> output = loss(input1, input2)
    """

    def __init__(self, dim=1, eps=1e-8, reduction='mean'):
        super(CosineLoss, self).__init__()
        self.dim = dim
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        output = F.cosine_similarity(input, target, self.dim, self.eps)

        if self.reduction == 'mean':
            output_reduced = output.mean()
        else:
            raise RuntimeError(f"Reduction: {self.reduction} does not support.")
        return output_reduced

class CosineLoss2(Module):
    r"""Returns cosine loss between :math:`input` and :math:`target`, computed along dim.

    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}.

    Args:
        dim (int, optional): Dimension where cosine similarity is computed. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`
        - Input2: :math:`(\ast_1, D, \ast_2)`, same shape as the Input1
        - Output: scalar
    Examples::
        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(100, 128)
        >>> loss = myloss.CosineLoss2(dim=1, eps=1e-6)
        >>> output = loss(input1, input2)
    """

    def __init__(self, dim=1, eps=1e-8, reduction='mean'):
        super(CosineLoss2, self).__init__()
        self.dim = dim
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if(input.shape != target.shape):
            raise RuntimeError(f"Shape of two tensors do not meet. input = {input.shape} while target = {target.shape}")

        dimension = input.shape[self.dim]

        dot_product = torch.bmm(input.view(-1, 1, dimension), target.view(-1, dimension, 1)).view(-1)
        norm1 = torch.norm(input, dim=self.dim)
        norm2 = torch.norm(target, dim=self.dim)

        output = dot_product / torch.max(torch.stack((norm1 * norm2, torch.ones(norm1.shape) * self.eps), dim=1), dim=1)[0]
        
        if self.reduction == 'mean':
            output_reduced = output.mean()
        else:
            raise RuntimeError(f"Reduction: {self.reduction} does not support.")
        return output_reduced

class CosineLoss3(Module):
    r"""Returns cosine loss between :math:`input` (A 2D vector represent a point on the complex plane) and :math:`target` (Target angle in radius).

    .. math ::
        \text{loss} = cos(\theta_{\text{target}} - arg(input)).

    Args:
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input1: :math:`(\ast_1, 2)`
        - Input2: :math:`(\ast_1, 1)`
        - Output: scalar
    Examples::
        >>> input1 = torch.randn(100, 2)
        >>> input2 = torch.randn(100, 1)
        >>> loss = myloss.CosineLoss3(eps=1e-6)
        >>> output = loss(input1, input2)
    """

    def __init__(self, eps=1e-8, reduction='mean'):
        super(CosineLoss3, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        if(len(input.shape) > 2):
            if(input.shape[1] == 1 or input.shape[2] == 1):
                input = input.view(-1, 2)
        
        if(len(target.shape) > 2):
            if(target.shape[1] == 1 or target.shape[2] == 1):
                target = target.view(-1, 2)

        if(len(input.shape) > 2 or len(target.shape) > 2 or input.shape[1] != 2 or target.shape[1] != 1):
            raise RuntimeError(f"Input should be a 2D vector and target should be a scalar but receive input = {input.shape} and target = {target.shape}")

        re_input, im_input = torch.split(input, 1, dim=1)
        arg_input = torch.atan2(re_input, im_input)
        
        output = torch.cos(target - arg_input)
        
        if self.reduction == 'mean':
            output_reduced = output.mean()
        else:
            raise RuntimeError(f"Reduction: {self.reduction} does not support.")
        return output_reduced

if __name__=="__main__":
    import numpy as np
    from torch.autograd import Variable
    from torch.nn import CosineSimilarity

    cos = CosineSimilarity()
    loss1 = CosineLoss(reduction="mean")
    loss2 = CosineLoss2(reduction="mean")
    loss3 = CosineLoss3(reduction="mean")

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

    input3 = torch.Tensor([[np.arctan2(1, 2)],
                          [np.arctan2(3, 4)],
                          [np.arctan2(5, 2)],
                          [np.arctan2(2, 3)],
                          [np.arctan2(4, 5)]])

    output = cos(input1, input2).mean()
    output.backward()
    print(output)
    print(input1.grad)
    input1.grad.zero_()

    output1 = loss1(input1, input2)
    output1.backward()
    print(output1)
    print(input1.grad)
    input1.grad.zero_()

    output2 = loss2(input1, input2)
    output2.backward()
    print(output2)
    print(input1.grad)
    input1.grad.zero_()

    output3 = loss3(input1, input3)
    output3.backward()
    print(output3)
    print(input1.grad)
    input1.grad.zero_()

    assert output1 == output2