import torch
import torch.nn.functional as F
from torch import nn
from module.resnet import resnet50
from torchsummary import summary

BatchNorm2d = nn.BatchNorm2d


class Resnet(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(pretrained)

        if dilate_scale == 8:
            model.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(
                partial(self._nostride_dilate, dilate=2))
        # 使用预训练的resnet，除了Avgpool和FC
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu1 = model.relu1
        self.conv2 = model.conv2
        self.bn2 = model.bn2
        self.relu2 = model.relu2
        self.conv3 = model.conv3
        self.bn3 = model.bn3
        self.relu3 = model.relu3
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_aux = self.layer3(x_2)
        x = self.layer4(x_aux)
        return x_1, x_2, x_aux, x
class FSPBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(FSPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.bn1 = BatchNorm2d(midplanes)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.bn2 = BatchNorm2d(midplanes)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.conv3 = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        _, _, h, w = x.size()

        identity = x
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        #x1 = x1.expand(-1, -1, h, w)
        #        x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        #x2 = x2.expand(-1, -1, h, w)
        #        x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 * x2)
        x = self.conv3(x).sigmoid()
        x = torch.mul(x, identity)
        return x

class PPMModule(nn.Module):
    def __init__(self, in_channels):
        super(PPMModule, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels//4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))

        self.bottleneck = nn.Sequential(nn.Conv2d((in_channels*2), out_channels, kernel_size=3, padding=1, bias=False),
                                        norm_layer(out_channels), nn.ReLU(inplace=True),
                                        #nn.Dropout2d(0.1),
                                         )
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), norm_layer(out_channels), nn.ReLU(True))
        self.fsp = FSPBlock(out_channels, out_channels)

        # bilinear interpolate options

    def forward(self, x):
        _, _, h, w = x.size()
        identity = self.conv(x)
        identity = self.fsp(identity)
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode='bilinear', align_corners=True)
        feat1 = feat1 + identity
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode='bilinear', align_corners=True)
        feat2 = feat2 + identity
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode='bilinear', align_corners=True)
        feat3 = feat3 + identity
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode='bilinear', align_corners=True)
        feat4 = feat4 + identity

        out = torch.cat((x, feat1, feat2, feat3, feat4), dim=1)
        out = self.bottleneck(out)
        return out

class CAM(nn.Module):
    def __init__(self, inplanes):
        super(CAM, self).__init__()

        self.channel_in = inplanes
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C,  height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class Spatial_branch(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(Spatial_branch, self).__init__()

        self.convblock1 = nn.Sequential(nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
                                        nn.BatchNorm2d(outchannels),
                                        nn.ReLU(inplace=True),)
        self.conv3X1 = nn.Sequential(
            nn.Conv2d(inchannels * 2, outchannels, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True), )
        self.conv1X3 = nn.Sequential(
            nn.Conv2d(inchannels * 2, outchannels, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True), )
        self.cam = CAM(outchannels)

    def forward(self, x):

        x = self.convblock1(x)
        x = self.conv3X1(x)
        x = self.conv1X3(x)
        x = self.cam(x)

        return x


class MyNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone="resnet50", pretrained=True, aux_branch=True):
        super(MyNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone == "resnet50":
            self.backbone = Resnet(downsample_factor, pretrained)
            aux_channel = 1024
            out_channel = 2048
        else:
            raise ValueError('Unsupported backbone - `{}`, Use resnet50, resnet101.'.format(backbone))

        self.master_branch = nn.Sequential(PPMModule(out_channel),)

        self.conv = nn.Sequential(nn.Conv2d(aux_channel, out_channel // 8, kernel_size=3, padding=1, bias=False),
                                  norm_layer(out_channel // 8), nn.ReLU(inplace=True),
                                  nn.Dropout2d(0.1),
                                  nn.Conv2d(out_channel // 8, num_classes, kernel_size=1),)

        self.spatial_branch = Spatial_branch(256,512)
        self.aux_branch = aux_branch
        if self.aux_branch:
            self.auxiliary_branch = nn.Sequential(
                nn.Conv2d(aux_channel, out_channel // 8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel // 8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(out_channel // 8, num_classes, kernel_size=1),)

            self.auxiliary_branch2 = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, kernel_size=1), )

        self.initialize_weights(self.master_branch)

        self.initialize_weights(self.spatial_branch)
        self.initialize_weights(self.conv)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_1, x_2, x_aux, x = self.backbone(x)
        _, _, h, w = x.size()
        output = self.master_branch(x)
        output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=True)
        output_cam = self.spatial_branch(x_1)
#        output = output + output_cam
        output = torch.cat((output, output_cam), 1)
        output = self.conv(output)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)

        if self.aux_branch:
            output_aux = self.auxiliary_branch(x_aux)
            output_aux2 = self.auxiliary_branch2(x_2)
            # 辅助分支的输出upsample到输入图像的大小，和真实值进行比较，做loss
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            output_aux2 = F.interpolate(output_aux2, size=input_size, mode='bilinear', align_corners=True)

            return output_aux, output, output_aux2
        else:
            return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.fill_(1e-5)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0005)
                    m.bias.data.zero_()

#net = MyNet(num_classes=21, downsample_factor=8, backbone='resnet50', pretrained=False, aux_branch=False).cuda()
#summary(net, (3, 480, 480))

