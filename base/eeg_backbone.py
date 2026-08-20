import torch.nn as nn
import numpy as np
import torch


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class EEGProjectLayer(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3):
        super(EEGProjectLayer, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
                                   ResidualAdd(nn.Sequential(
                                       nn.GELU(),
                                       nn.Linear(proj_dim, proj_dim),
                                       nn.Dropout(drop_proj),
                                   )),
                                   nn.LayerNorm(proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


class ColorHead(nn.Module):
    def __init__(self, input_dim=1440, hidden_dims=[512, 256], output_dim=13):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class EEGProjectLayerColour(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3):
        super(EEGProjectLayerColour, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
                                   ResidualAdd(nn.Sequential(
                                       nn.GELU(),
                                       nn.Linear(proj_dim, proj_dim),
                                       nn.Dropout(drop_proj),
                                   )),
                                   nn.LayerNorm(proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        self.colour_head = ColorHead(input_dim=proj_dim)

    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        x_colour = self.colour_head(x)
        return x, x_colour


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class BaseModel(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, embedding_dim=1440):
        super(BaseModel, self).__init__()

        self.backbone = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(z_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x


class Shallownet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )


class Deepnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=1400)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),

            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )


class EEGnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim=1248)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5)
        )


class TSconv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )


class EEGConvProjectLayerColour(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3, f1=16, f2=32, f3=64, k1=25, k2=15, colour_hidden_dims=(512, 256)):
        super().__init__()

        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

        # Required because the convolutions below are grouped.
        if f2 % f1 != 0:
            raise ValueError(
                f"f2 ({f2}) must be divisible by f1 ({f1})"
            )

        if f3 % f2 != 0:
            raise ValueError(
                f"f3 ({f3}) must be divisible by f2 ({f2})"
            )

        self.encoder = nn.Sequential(


            # 1. Temporal feature extraction
            nn.Conv2d(in_channels=1, out_channels=f1, kernel_size=(1, k1), padding=(0, 12), bias=False),
            nn.BatchNorm2d(f1),

            # 2. Spatial filtering across all EEG channels
            nn.Conv2d(in_channels=f1, out_channels=f2, kernel_size=(c_num, 1), groups=f1, bias=False),

            nn.BatchNorm2d(f2),
            nn.ELU(),

            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(drop_proj),

            # 3. Additional temporal feature extraction
            nn.Conv2d(in_channels=f2, out_channels=f3, kernel_size=(1, k2), padding=(0, 7), groups=f2, bias=False),

            nn.Conv2d(in_channels=f3, out_channels=f3, kernel_size=(1, 1), bias=False),

            nn.BatchNorm2d(f3),
            nn.ELU(),

            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),

            nn.Dropout(drop_proj),

            nn.AdaptiveAvgPool2d((1, 8)),
        )

        # Encoder output:
        # (batch, f3, 1, 8)
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(f3 * 8, z_dim,), nn.GELU(), nn.Dropout(drop_proj), nn.LayerNorm(z_dim),)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.softplus = nn.Softplus()

        self.colour_head = ColorHead(input_dim=z_dim, hidden_dims=colour_hidden_dims)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                "Expected EEG input with shape "
                f"(batch, channels, time), got {x.shape}"
            )

        if x.shape[1] != self.c_num:
            raise ValueError(
                f"Expected {self.c_num} EEG channels, "
                f"but received {x.shape[1]}"
            )

        x = x.unsqueeze(1)

        x = self.encoder(x)
        x = self.projection(x)

        x_colour = self.colour_head(x)

        return x, x_colour

