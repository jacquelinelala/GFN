import torch
import torch.nn as nn
import math
import torch.nn.init as init


class _Residual_Block_DB(nn.Module):
    def __init__(self, num_ft):
        super(_Residual_Block_DB, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=num_ft, out_channels=num_ft, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output

class _Residual_Block_SR(nn.Module):
    def __init__(self):
        super(_Residual_Block_SR, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output = torch.add(output, identity_data)
        return output


class _EDNet(nn.Module):
    def __init__(self):
        super(_EDNet, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual_db = self.make_layer(_Residual_Block_DB, 6, 64)

        self.conv_stride1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.residual_db1 = self.make_layer(_Residual_Block_DB, 6, 128)

        self.conv_stride2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.residual_db2 = self.make_layer(_Residual_Block_DB, 6, 256)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3, bias=True)
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer, num_ft):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_ft))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        res1 = out
        out = self.residual_db(out)
        out = torch.add(out, res1)
        out = self.conv_stride1(out)
        residual1 = out
        out = self.residual_db1(out)
        out = torch.add(out, residual1)
        out = self.conv_stride2(out)
        residual2 = out
        out = self.residual_db2(out)
        out = torch.add(out, residual2)
        out = self.deconv2(out)
        out = self.deconv1(out)
        res_ft = out
        out = torch.add(out, res1)
        out = self.conv_out(out)
        return res_ft, out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual1 = self.make_layer(_Residual_Block_SR, 8)
        self.residual2 = self.make_layer(_Residual_Block_SR, 8)

        self.conv_mid = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),  # eliminate the artifacts
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))
        )

        self.ednet = self.make_layer(_EDNet, 1)

        ##############Attention Model#####################
        self.attention_model = nn.Sequential(
            nn.Conv2d(in_channels=131, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        [detail_ft, lr_deblur] = self.ednet(x)
        att_in1 = detail_ft

        out = self.conv_input(x)
        residual = out
        out = self.residual1(out)
        out = self.conv_mid(out)
        out = torch.add(out, residual)
        att_in2 = out
        att_in = torch.cat((x,att_in1, att_in2), dim=1)
        scoremap = self.attention_model(att_in)
        detail_ft = torch.mul(detail_ft, scoremap)

        out = torch.add(out, detail_ft)

        out = self.residual2(out)
        out = self.upscale4x(out)
        sr = self.conv_output(out)
        return lr_deblur, scoremap, sr