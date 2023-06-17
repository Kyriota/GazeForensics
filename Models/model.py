import torch.nn as nn
import torch
from resnet import resnet18, resnet50



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

    def forward(self, x):
        for i in range(len(self.fc)):
            x = self.fc[i](x)
            if i != len(self.fc) - 1:
                x = nn.ReLU()(x)
        return x



class GazeForensics(nn.Module):
    def __init__(self, leaky=0, save_mid_layer=False, save_last_layer=False):
        nn.Module.__init__(self)

        self.leaky = leaky
        self.save_last_layer = save_last_layer

        self.base_model = resnet18(pretrained=True, save_mid_layer=save_mid_layer)
        self.base_model.fc2 = nn.Linear(1000, 256 + leaky)

        self.MHA_Q_fc = FC_block(512 + leaky, 256, mid_sizes=[320])
        self.MHA_K_fc = FC_block(512 + leaky, 256, mid_sizes=[320])
        self.MHA_V_fc = FC_block(512 + leaky, 256, mid_sizes=[320])
        self.multihead_attn = nn.MultiheadAttention(256, 4, batch_first=True)

        self.last_fc = FC_block(256, 2, mid_sizes=[128, 64])

        self.activation = nn.ReLU()

        self.IOP = {}


    def forward(self, input):
        backendOut = self.base_model(input.view((input.size(0) * 14, 3) + input.size()[-2:]))
        if self.save_last_layer:
            self.IOP[self.base_model] = backendOut[:, :512]
        backendOut = backendOut.view(input.size(0), 14, 256 + self.leaky)
        backendOut = self.activation(backendOut)
        
        attn_q = self.MHA_Q_fc(backendOut)
        attn_k = self.MHA_K_fc(backendOut)
        attn_v = self.MHA_V_fc(backendOut)
        attnOut, _ = self.multihead_attn(attn_q, attn_k, attn_v)
        attnOut = attnOut.reshape(input.size(0) * 14, 256)
        output = self.last_fc(attnOut)
        output = output.reshape(input.size(0), 14, 2)
        output = torch.mean(output, dim=1)
        output = torch.softmax(output, dim=1)

        return output