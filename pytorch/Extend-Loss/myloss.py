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

if __name__=="__main__":
    from torch.autograd import Variable
    from torch.nn import CosineSimilarity

    cos = CosineSimilarity()
    loss1 = CosineLoss(reduction="mean")
    loss2 = CosineLoss2(reduction="mean")

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

    assert output1 == output2