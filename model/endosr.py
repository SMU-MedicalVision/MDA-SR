'''
'''
import torch.nn as nn


def make_model(args, parent=False):
    return EndoSR(args)


class Degradation_model(nn.Module):
    def __init__(self, in_channels, out_channels, n_feats=32, n_blocks=8, scale=2, bias=True):
        super(Degradation_model, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channels, n_feats, kernel_size=3, stride=1, padding=1), nn.ReLU(True))
        body = [ResBlock_D(in_channels=n_feats, out_channels=n_feats, bias=bias) for _ in range(n_blocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(nn.Conv2d(n_feats, out_channels, kernel_size=scale, stride=scale))

    def forward(self, x):
        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)

        return x


class ResBlock_D(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super(ResBlock_D, self).__init__()
        self.Block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.Block(x) + x
        return x


class ResBlock_R(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, res_scale=1, bias=True):
        super(ResBlock_R, self).__init__()
        self.res_scale = res_scale
        layer = []
        layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        layer.append(nn.ReLU(True))
        layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.res = nn.Sequential(*layer)

    def forward(self, x):
        return x + self.res(x) * self.res_scale


class Encode_model(nn.Module):
    def __init__(self, in_channels, n_blocks=8, scale=2):
        super(Encode_model, self).__init__()
        if scale == 2:
            res_scale = 1
            n_feats = 64
        elif scale == 4:
            res_scale = 0.1
            n_feats = 128
        elif scale == 8:
            res_scale = 0.1
            n_feats = 128
        self.head = nn.Sequential(nn.Conv2d(in_channels, n_feats, kernel_size=3, stride=1, padding=1), nn.ReLU(True))
        body = [ResBlock_R(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale) for _ in range(n_blocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)

        return x


class Reconstruction_model(nn.Module):
    def __init__(self, out_channels, n_blocks=8, scale=2):
        super(Reconstruction_model, self).__init__()
        if scale == 2:
            res_scale = 1
            n_feats = 64
            self.tail = nn.Sequential(nn.Conv2d(n_feats, n_feats*4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1))
        elif scale == 4:
            res_scale = 0.1
            n_feats = 128
            self.tail = nn.Sequential(nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1)
                                      )
        elif scale == 8:
            res_scale = 0.1
            n_feats = 128
            self.tail = nn.Sequential(nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, stride=1, padding=1),
                                      nn.PixelShuffle(2),
                                      nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1)
                                      )
        # self.head = nn.Sequential(nn.Conv2d(in_channels, n_feats, kernel_size=3, stride=1, padding=1), nn.ReLU(True))
        body = [ResBlock_R(in_channels=n_feats, out_channels=n_feats, res_scale=res_scale) for _ in range(n_blocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        # x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class EndoSR(nn.Module):
    def __init__(self, args):
        super(EndoSR, self).__init__()
        self.scale = int(args.scale)
        self.encode = Encode_model(in_channels=3, n_blocks=8, scale=self.scale)
        self.recon = Reconstruction_model(out_channels=3, n_blocks=8, scale=self.scale)

    def forward(self, x):
        feature = self.encode(x)
        img = self.recon(feature)
        return feature, img


# Defines the PatchGAN discriminator with the specified arguments.
class Discriminator_model(nn.Module):
    def __init__(self, scale, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(Discriminator_model, self).__init__()
        if scale == 2:
            ndf = 64
        elif scale == 4:
            ndf = 128
        elif scale == 8:
            ndf = 128

        use_bias = False
        kw = 4
        padw = 1
        sequence = []

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                         norm_layer(ndf * nf_mult),
                         nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = 2
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                     norm_layer(ndf * nf_mult),
                     nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
