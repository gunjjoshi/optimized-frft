import torch
import torch.nn as nn
import math

class FRFT(nn.Module):
    def __init__(self, in_channels, order=0.5, H=64, W=64):
        super(FRFT, self).__init__()
        # Ensure in_channels is at least 3 for the division
        assert in_channels >= 3, "in_channels must be at least 3"
        
        C0 = in_channels // 3
        C1 = in_channels - 2 * C0

        self.conv_0 = nn.Conv2d(C0, C0, kernel_size=3, padding=1)
        self.conv_05 = nn.Conv2d(2 * C1, 2 * C1, kernel_size=1, padding=0)
        self.conv_1 = nn.Conv2d(2 * C0, 2 * C0, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Trainable fractional order (bounded between 0 and 1)
        self.order = nn.Parameter(torch.sigmoid(torch.randn(1)))
        
        # Initialize matrices as buffers
        self.register_buffer('h_test', None)
        self.register_buffer('w_test', None)
        
        # Modified channel attention mechanism
        reduction_ratio = 16
        reduced_channels = max(in_channels // reduction_ratio, 1)  # Ensure at least 1 channel
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        # Initialize FRFT matrices
        self.initialize_frft_matrices(H, W)

    def initialize_frft_matrices(self, H, W):
        """Initialize FRFT matrices after model is moved to the correct device"""
        self.h_test = self.dfrtmtrx(H, self.order)
        self.w_test = self.dfrtmtrx(W, self.order)

    def dfrtmtrx(self, N, a):
        """Compute FRFT matrix using the current device"""
        device = self.order.device
        
        # Create identity matrix on the correct device
        Evec = torch.eye(N, device=device, dtype=torch.complex64)
        
        even = 1 - (N % 2)
        l = torch.arange(N - 1 + even, device=device)
        
        # Create diagonal matrix directly on the correct device
        f = torch.exp(-1j * math.pi / 2 * a * l).diag()
        
        # Compute FRFT matrix
        F = N ** 0.5 * torch.einsum('ij,jk,ni->nk', f, Evec.T, Evec)
        return F

    def FRFT2D(self, matrix, h_test, w_test):
        """Compute 2D FRFT"""
        N, C, H, W = matrix.shape
        
        # Expand matrices for batch processing
        h_test_expanded = h_test.unsqueeze(0).expand(N, -1, -1)
        w_test_expanded = w_test.unsqueeze(0).expand(N, -1, -1)
        
        # Convert to complex and apply fftshift
        matrix = torch.fft.fftshift(matrix.to(dtype=torch.complex64), dim=(2, 3))
        
        # Apply FRFT using optimized einsum operations
        out = torch.einsum('nchw,nhw->nchw', matrix, h_test_expanded)
        out = torch.einsum('nchw,nwh->nchw', out, w_test_expanded)
        
        return torch.fft.fftshift(out, dim=(2, 3))

    @torch.autocast(device_type='cuda', dtype=torch.float16)  # Updated autocast syntax
    def forward(self, x):
        # Print shape for debugging
        batch_size, channels, height, width = x.shape
        print(f"Input shape: {x.shape}")
        
        # Ensure matrices are initialized on the correct device
        if self.h_test is None or self.w_test is None:
            self.initialize_frft_matrices(height, width)
        
        # Apply attention and FRFT
        attention_weights = self.channel_attention(x)
        print(f"Attention weights shape: {attention_weights.shape}")
        
        x = x * attention_weights
        return self.FRFT2D(x, self.h_test, self.w_test)