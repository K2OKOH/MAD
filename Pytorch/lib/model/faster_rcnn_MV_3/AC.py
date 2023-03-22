import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Encoder
            # input (b, 512, 40, 76)
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
            # output (b, 128, 10, 19)
        )

    def forward(self, *input):
        out = self.encoder(*input)
        return out

class AutoDecoder(nn.Module):
    def __init__(self):
        super(AutoDecoder, self).__init__()

        self.decoder = nn.Sequential(
        # DEcoder
        nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.ConvTranspose2d(512, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU()

        )

    def forward(self, *input):
        out = self.decoder(*input)
        return out

class ImgEncoder_1(nn.Module):
    def __init__(self):
        super(ImgEncoder_1, self).__init__()
        
        self.encoder = nn.Sequential(
            # Encoder
            # input (b, 512, 40, 76)
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
            # output (b, 128, 10, 19)
        )

    def forward(self, *input):
        out = self.encoder(*input)
        return out

class ImgDecoder_1(nn.Module):
    def __init__(self):
        super(ImgDecoder_1, self).__init__()

        self.decoder = nn.Sequential(
        # DEcoder
        nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.ConvTranspose2d(512, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.ReLU()

        )

    def forward(self, *input):
        out = self.decoder(*input)
        return out

class ImgEncoder_2(nn.Module):
    def __init__(self):
        super(ImgEncoder_2, self).__init__()
        
        self.encoder = nn.Sequential(
            # Encoder
            # input (b, 512, 40, 76)
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=2, bias=False, dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU()
            # output (b, 128, 10, 19)
        )

    def forward(self, *input):
        out = self.encoder(*input)
        return out

class ImgDecoder_2(nn.Module):
    def __init__(self):
        super(ImgDecoder_2, self).__init__()

        self.decoder = nn.Sequential(
        # DEcoder
        nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=2, output_padding=1, bias=False, dilation=2),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.ConvTranspose2d(512, 1024, kernel_size=3, stride=2, padding=2, output_padding=1, bias=False, dilation=2),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=2, bias=False, dilation=2),
        nn.ReLU()

        )

    def forward(self, *input):
        out = self.decoder(*input)
        return out

class ImgEncoder_3(nn.Module):
    def __init__(self):
        super(ImgEncoder_3, self).__init__()
        
        self.encoder = nn.Sequential(
            # Encoder
            # input (b, 512, 40, 76)
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3, bias=False, dilation=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=3, bias=False, dilation=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=3, bias=False, dilation=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=3, bias=False, dilation=3),
            nn.BatchNorm2d(128),
            nn.ReLU()
            # output (b, 128, 10, 19)
        )

    def forward(self, *input):
        out = self.encoder(*input)
        return out

class ImgDecoder_3(nn.Module):
    def __init__(self):
        super(ImgDecoder_3, self).__init__()

        self.decoder = nn.Sequential(
        # DEcoder
        nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=3, output_padding=1, bias=False, dilation=3),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=3, bias=False, dilation=3),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.ConvTranspose2d(512, 1024, kernel_size=3, stride=2, padding=3, output_padding=1, bias=False, dilation=3),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=1, padding=3, bias=False, dilation=3),
        nn.ReLU()

        )

    def forward(self, *input):
        out = self.decoder(*input)
        return out

class InsEncoder(nn.Module):
    def __init__(self):
        super(InsEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
        # Encoder
        # input (b, 4096)
        nn.Linear(4096, 2048),
        nn.ReLU(True),
        nn.Linear(2048, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 512)
        # input (b, 512)
        )

    def forward(self, *input):
        out = self.encoder(*input)
        return out

class InsDecoder(nn.Module):
    def __init__(self):
        super(InsDecoder, self).__init__()

        self.decoder = nn.Sequential(
        # DEcoder
        nn.Linear(512, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 2048),
        nn.ReLU(True),
        nn.Linear(2048, 4096)
        )

    def forward(self, *input):
        out = self.decoder(*input)
        return out
