import torch
import torch.nn as nn


class UNetMini(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_channels=16, depth=3):
        """
        Flexible U-Net architecture.
        in and out channels can be specified, as well as the number of feature channels and depth of the network.
        """
        super().__init__()
        self.depth = depth

        # Helper Class for Double Convolution Block
        # This block consists of two convolutional layers, each followed by batch normalization and ReLU activation.
        class DoubleConv(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(
                        in_c, out_c, 3, padding=1
                    ),  # 3x3 convolution with padding to maintain spatial dimensions
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(
                        inplace=True
                    ),  # inplace=True saves memory by modifying the input directly
                    nn.Conv2d(out_c, out_c, 3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.conv(x)

        # Lists for the layers
        self.encoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(2)

        # ENCODER (Down)
        # current_in and current_out are used to track the number of channels as we go deeper into the network.
        current_in = in_channels
        current_out = feature_channels

        # We create depth many encoder blocks
        for i in range(depth):
            self.encoders.append(DoubleConv(current_in, current_out))
            current_in = current_out
            current_out *= 2  # Channels double with each depth

        # BOTTLENECK
        # The deepest point (connection between encoder and decoder)
        self.bottleneck = DoubleConv(current_in, current_out)

        # DECODER (Up)
        for i in range(depth):
            # 1. Upsampling (Transpose Conv)
            # Input: current_out, Output: current_in (Halving channels)
            self.upconvs.append(
                nn.ConvTranspose2d(
                    current_out, current_in, kernel_size=2, stride=2
                )  # 2x2 transpose convolution with stride 2 to double the spatial dimensions
            )

            # 2. Conv Block after Concatenation
            # Due to skip connections, input = current_in + current_in (from encoder)
            self.decoders.append(DoubleConv(current_in * 2, current_in))

            current_out = current_in
            current_in //= 2  # Halving channels as we move up the decoder

        # FINAL OUTPUT
        # The final layer reduces the number of channels to the desired output channels (e.g., 1 for binary segmentation).
        self.out_conv = nn.Conv2d(feature_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Store skip connections for the decoder
        skips = []

        # Encoder path
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)  # save for skip connection
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        # We use 'zip' to iterate over upsamplers, decoders, and skip connections in parallel
        # skips[::-1] reverses the list so we get the most appropriate element first
        for up, dec, skip in zip(self.upconvs, self.decoders, skips[::-1]):
            x = up(x)

            # Concatenate (Add skip connection)
            x = torch.cat([x, skip], dim=1)

            x = dec(x)
        # Final output layer with sigmoid activation for binary output
        return torch.sigmoid(self.out_conv(x))
