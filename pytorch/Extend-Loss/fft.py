import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import numpy.fft as fft

class Fft(autograd.Function):

    @staticmethod
    def forward(self, X_re, X_im):
        X_re, X_im = X_re.detach().numpy(), X_im.detach().numpy()
        X_complex = X_re + X_im * 1j
        Y_complex = fft.fft(X_complex)
        Y_re, Y_im = Y_complex.real, Y_complex.imag
        Y_re, Y_im = Variable(torch.FloatTensor(Y_re)), Variable(torch.FloatTensor(Y_im))
        return Y_re, Y_im
    
    @staticmethod
    def backward(self, Y_re_grad, Y_im_grad):
        Y_re_grad, Y_im_grad = Y_re_grad.detach().numpy(), Y_im_grad.detach().numpy()
        Y_grad_complex = Y_re_grad + Y_im_grad * 1j
        X_grad_complex = fft.fft(Y_grad_complex)
        X_re_grad, X_im_grad = X_grad_complex.real, X_grad_complex.imag
        X_re_grad, X_im_grad = Variable(torch.FloatTensor(X_re_grad)), Variable(torch.FloatTensor(X_im_grad))
        return X_re_grad, X_im_grad

if __name__=="__main__":
    import numpy as np
    from torch.autograd import Variable

    mfft = Fft.apply

    input0 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]) + 0j # Impluse function
    input1 = np.exp(2j * np.pi * np.arange(8) / 8)
    input2 = np.exp(2j * np.pi * np.arange(8) / 4)
    input3 = input1 + input2 # To test FFT(t1+t2) = FFT(t1) + FFT(t2)

    re0 = Variable(torch.FloatTensor(input0.real), requires_grad=True)
    im0 = Variable(torch.FloatTensor(input0.imag), requires_grad=True)

    re1 = Variable(torch.FloatTensor(input1.real), requires_grad=True)
    im1 = Variable(torch.FloatTensor(input1.imag), requires_grad=True)
    
    re2 = Variable(torch.FloatTensor(input2.real), requires_grad=True)
    im2 = Variable(torch.FloatTensor(input2.imag), requires_grad=True)
    
    re3 = Variable(torch.FloatTensor(input3.real), requires_grad=True)
    im3 = Variable(torch.FloatTensor(input3.imag), requires_grad=True)

    print("INPUT0 RE 0: ", re0)
    print("INPUT0 IM 0: ", im0)
    print()

    print("INPUT1 RE: ", re1)
    print("INPUT1 IM: ", im1)
    print()

    print("INPUT2 RE: ", re2)
    print("INPUT2 IM: ", im2)
    print()

    print("INPUT1+INPUT2 RE 3: ", re3)
    print("INPUT1+INPUT2 IM 3: ", im3)
    print()

    print()

    f_re0, f_im0 = mfft(re0, im0) # Will get uniformly 1 over all freq
    f_re1, f_im1 = mfft(re1, im1) # FFT(t1)
    f_re2, f_im2 = mfft(re2, im2) # FFT(t2)
    f_re3, f_im3 = mfft(re3, im3) # FFT(t1+t2)
    f_re3_, f_im3_ = f_re1 + f_re2, f_im1 + f_im2 # FFT(t1) + FFT(t2)

    print("FFT0 RE: ", f_re0)
    print("FFT0 IM: ", f_im0)
    print()

    print("FFT1 RE 1: ", f_re1)
    print("FFT1 IM 1: ", f_im1)
    print()

    print("FFT2 RE: ", f_re2)
    print("FFT2 IM: ", f_im2)
    print()

    print("FFT(1+2) RE: ", f_re3)
    print("FFT(1+2) IM: ", f_im3)
    print()

    print("FFT1+FFT2 RE: ", f_re3_)
    print("FFT1+FFT2 IM: ", f_im3_)
    print()

    print()

    f_mag1 = torch.sqrt(f_re1.pow(2) + f_im1.pow(2))
    f_mag2 = torch.sqrt(f_re2.pow(2) + f_im2.pow(2))
    f_mag3 = torch.sqrt(f_re3.pow(2) + f_im3.pow(2))
    f_mag3_ = torch.sqrt(f_re3_.pow(2) + f_im3_.pow(2))

    print("FFT1 MAG: ", f_mag1)
    print("FFT2 MAG: ", f_mag2)
    print("FFT(1+2) MAG: ", f_mag3)
    print("FFT1+FFT2 MAG: ", f_mag3_)

    print()

    f_mag1_reduce = f_mag1.mean()

    print("FFT MAG AVG: ", f_mag1_reduce)

    f_mag1_reduce.backward()

    print("GRAD RE: ", re1.grad)
    print("GRAD IM: ", im1.grad)