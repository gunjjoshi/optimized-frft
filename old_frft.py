import torch
import torch.nn as nn
import math

# Core module (Old FRFT implementation)
class FRFT(nn.Module):
    def __init__(self, in_channels, order=0.5):
        super(FRFT, self).__init__()
        C0 = int(in_channels / 3)
        C1 = int(in_channels) - 2 * C0
        self.conv_0 = nn.Conv2d(C0, C0, kernel_size=3, padding=1)
        self.conv_05 = nn.Conv2d(2 * C1, 2 * C1, kernel_size=1, padding=0)
        self.conv_1 = nn.Conv2d(2 * C0, 2 * C0, kernel_size=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.order = nn.Parameter(torch.randn(1))

    def dfrtmtrx(self, N, a):
        # Compute the FRFT transformation matrix
        app_ord = 2
        Evec = torch.eye(N).to(torch.complex64).cuda()
        even = 1 - (N % 2)
        l = torch.tensor(list(range(0, N - 1)) + [N - 1 + even]).cuda()
        f = torch.diag(torch.exp(-1j * math.pi / 2 * a * l))
        F = N ** (1 / 2) * torch.einsum("ij,jk,ni->nk", f, Evec.T, Evec)
        return F

    def FRFT2D(self, matrix):
        N, C, H, W = matrix.shape
        h_test = self.dfrtmtrx(H, self.order).cuda()
        w_test = self.dfrtmtrx(W, self.order).cuda()
        h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=C, dim=0)
        h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=N, dim=0)
        w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=C, dim=0)
        w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=N, dim=0)

        matrix = torch.fft.fftshift(matrix, dim=(2, 3)).to(dtype=torch.complex64)
        out = torch.matmul(h_test, matrix)
        out = torch.matmul(out, w_test)
        return torch.fft.fftshift(out, dim=(2, 3))

    def forward(self, x):
        return self.FRFT2D(x)
