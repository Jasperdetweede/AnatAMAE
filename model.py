import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        """
        Convolutional autoencoder.
        Input: 1x256x256 -> Encoder -> latent features -> Decoder.
        """
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),   # 256 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 128 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),

            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 64 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 32 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),

            nn.Conv2d(256, latent_dim, 4, stride=2, padding=1), # 16 -> 8
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(inplace=True),
        )
        # flatten point
        self._latent_spatial = 8  # final spatial dims: 256 -> 8 after 5×2 strides
        self.latent_dim = latent_dim

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, stride=2, padding=1), # 8 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),       # 16 -> 32
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),        # 32 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),         # 64 -> 128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32),

            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),          # 128 -> 256
            nn.Tanh(),  # reconstruct intensities in [-1,1]
        )

    def forward(self, x):
        """
        Args:
            x: FloatTensor [B,1,256,256]
        Returns:
            rec : [B,1,256,256] reconstructed input
        """
        z = self.encoder(x)   # [B,L,8,8]
        rec = self.decoder(z)  # [B,1,256,256]
        return rec

class Classifier(nn.Module):
    def __init__(self, encoder: nn.Module, latent_dim: int):
        """
        A classifier that takes encoder outputs of shape [B, latent_dim, H, W],
        does global pooling + MLP -> scalar probability.
        
        Args:
            encoder: nn.Module mapping x -> features [B, latent_dim, h, w]
            latent_dim: number of channels in encoder output
        """
        super().__init__()
        self.encoder = encoder

        # global pooling -> flatten -> dropout -> linear -> sigmoid
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, latent_dim, 1, 1]
            nn.Flatten(),                  # [B, latent_dim]
            nn.Dropout(0.5),
            nn.Linear(latent_dim, 1),      # [B, 1]
            nn.Sigmoid(),                  # [B, 1] in (0,1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x -> encoder  ->  classifier  ->  [B] probability
        """
        feats = self.encoder(x)             # [B, latent_dim, H, W]
        out   = self.classifier(feats)      # [B, 1]
        return out.squeeze(1)               # [B]