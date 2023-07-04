import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F


class FC_block(nn.Module):
    def __init__(self, input_size, output_size, mid_sizes=[]):
        # mid_sizes: list of int, the size for each hidden layer
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.mid_sizes = mid_sizes

        self.fc = nn.ModuleList()
        if len(mid_sizes) == 0:
            self.fc.append(nn.Linear(input_size, output_size))
        else:
            self.fc.append(nn.Linear(input_size, mid_sizes[0]))
            for i in range(len(mid_sizes) - 1):
                self.fc.append(nn.Linear(mid_sizes[i], mid_sizes[i + 1]))
            self.fc.append(nn.Linear(mid_sizes[-1], output_size))
        
        # Initialize linear layers with He initialization
        self._init_weights()

    def _init_weights(self):
        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
            if i != len(self.fc) - 1:
                x = nn.ReLU()(x)
        return x


class L2CS(nn.Module):
    def __init__(self, block, layers, num_bins, eval_mode=False, emb_dim=0):
        self.eval_mode = eval_mode
        self.need_decoder = emb_dim != 0
        self.inplanes = 64
        super(L2CS, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.need_decoder:
            self.avg_encoder = FC_block(512 * block.expansion, emb_dim, [256, 256])
            self.avg_decoder = FC_block(emb_dim, 2048, [512, 1024])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        if eval_mode:
            self.fc_yaw_gaze = nn.Linear(2048, num_bins)
            self.fc_pitch_gaze = nn.Linear(2048, num_bins)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.need_decoder:
            x = self.avg_encoder(x)
            code = x.clone()
            x = self.avg_decoder(x)

        if self.eval_mode:
            pre_yaw_gaze =  self.fc_yaw_gaze(x)
            pre_pitch_gaze = self.fc_pitch_gaze(x)
            return pre_yaw_gaze, pre_pitch_gaze

        if self.need_decoder:
            return x, code
        return x