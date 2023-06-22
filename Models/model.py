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
    def __init__(self, leaky=0, save_backend_output=False, arch='resnet18'):
        nn.Module.__init__(self)

        self.leaky = leaky
        self.save_backend_output = save_backend_output

        if arch == 'resnet18':
            self.base_model = resnet18(pretrained=True)
            self.avg_size = 512
        elif arch == 'resnet50':
            self.base_model = resnet50(pretrained=True)
            self.avg_size = 2048
        else:
            raise NotImplementedError
        
        self.emb_dim = 2048
        self.MHA_dim = 256
        self.fpc = 14 # frames per clip
        self.base_fc = FC_block(self.avg_size, self.emb_dim + self.leaky, mid_sizes=[1024, 1024])

        self.MHA_Q_fc = FC_block(self.emb_dim + self.leaky, self.MHA_dim, mid_sizes=[320])
        self.MHA_K_fc = FC_block(self.emb_dim + self.leaky, self.MHA_dim, mid_sizes=[320])
        self.MHA_V_fc = FC_block(self.emb_dim + self.leaky, self.MHA_dim, mid_sizes=[320])
        self.multihead_attn = nn.MultiheadAttention(self.MHA_dim, 4, batch_first=True)

        self.last_fc = FC_block(self.MHA_dim, 2, mid_sizes=[128, 64])

        self.activation = nn.ReLU()

        self.backend_out = None


    def forward(self, input):
        backendOut = self.base_model(input.view((input.size(0) * self.fpc, 3) + input.size()[-2:]))
        backendOut = backendOut.view(input.size(0), self.fpc, self.avg_size)
        backendOut = self.activation(backendOut)
        backendOut = self.base_fc(backendOut)
        if self.save_backend_output:
            self.backend_out = backendOut
        backendOut = self.activation(backendOut)
        
        attn_q = self.MHA_Q_fc(backendOut)
        attn_k = self.MHA_K_fc(backendOut)
        attn_v = self.MHA_V_fc(backendOut)
        attnOut, _ = self.multihead_attn(attn_q, attn_k, attn_v)
        attnOut = attnOut.reshape(input.size(0) * self.fpc, self.MHA_dim)
        output = self.last_fc(attnOut)
        output = output.reshape(input.size(0), self.fpc, 2)
        output = output.mean(dim=1)
        output = torch.softmax(output, dim=1)

        return output