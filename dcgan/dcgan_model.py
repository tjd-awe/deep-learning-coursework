import torch.nn as nn


SUPPORTED_IMAGE_SIZES = (64, 128)


def validate_image_size(image_size):
    if image_size not in SUPPORTED_IMAGE_SIZES:
        raise ValueError(
            f"Unsupported imageSize={image_size}. Supported values are: {SUPPORTED_IMAGE_SIZES}"
        )


class Generator(nn.Module):
    def __init__(self, image_size, nz, ngf, nc):
        super(Generator, self).__init__()
        validate_image_size(image_size)

        layers = [
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        ]

        if image_size == 64:
            layers.extend([
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            ])
        else:
            extra_channels = max(ngf // 2, 1)
            layers.extend([
                nn.ConvTranspose2d(ngf, extra_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(extra_channels),
                nn.ReLU(True),
                nn.ConvTranspose2d(extra_channels, nc, 4, 2, 1, bias=False),
                nn.Tanh(),
            ])

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, image_size, ndf, nc):
        super(Discriminator, self).__init__()
        validate_image_size(image_size)

        layers = []
        in_channels = nc

        if image_size == 128:
            first_channels = max(ndf // 2, 1)
            layers.extend([
                nn.Conv2d(in_channels, first_channels, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = first_channels

        channel_schedule = [ndf, ndf * 2, ndf * 4, ndf * 8]

        for index, out_channels in enumerate(channel_schedule):
            layers.append(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False))
            if index != 0 or image_size == 128:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        layers.extend([
            nn.Conv2d(in_channels, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        ])

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
