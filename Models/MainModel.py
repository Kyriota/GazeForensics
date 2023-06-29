import torch
import torch.nn as nn

from Models.Resnet import resnet18
from Config import Config



def InitModel(config:Config):
    torch.manual_seed(config.model['seed'])
    model = GazeForensics(
        save_gaze_out=config.loss['gaze_weight'] > 0.0,
        leaky=config.model['leaky']
    )
    model = nn.DataParallel(model).cuda()
    return model



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
        
        # Initialize linear layers with Kaiming initialization
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



class GazeForensics(nn.Module):
    def __init__(self, save_gaze_out=False, leaky=0):
        nn.Module.__init__(self)

        self.save_gaze_out = save_gaze_out
        
        self.leaky = leaky
        self.avg_dim = 512
        self.emb_dim = 512 + leaky
        self.MHA_dim = 512
        self.MHA_comp_dim = 128
        self.fpc = 14 # frames per clip

        self.base_model = resnet18(pretrained=True)
        self.gaze_fc = FC_block(self.avg_dim, self.emb_dim, mid_sizes=[1024, 1024])

        self.MHA_Q_fc = FC_block(self.emb_dim, self.MHA_dim, mid_sizes=[768, 768])
        self.MHA_K_fc = FC_block(self.emb_dim, self.MHA_dim, mid_sizes=[768, 768])
        self.MHA_V_fc = FC_block(self.emb_dim, self.MHA_dim, mid_sizes=[768, 768])
        self.multihead_attn = nn.MultiheadAttention(self.MHA_dim, 4, batch_first=True)

        self.MHA_comp = FC_block(self.MHA_dim, self.MHA_comp_dim, mid_sizes=[512, 512])

        # Put addtional parts after fundamental parts initialized
        # So that within the same seed, most parts of the model are the same
        self.last_fc = FC_block(self.MHA_comp_dim * self.fpc, 3, mid_sizes=[512, 128])

        self.softmax = nn.Softmax(dim=1)

        self.gaze_out = None


    def forward(self, x):
        batch_size = x.size(0)
        x = self.base_model(x.view((batch_size * self.fpc, 3) + x.size()[-2:]))
        x = x.view(batch_size, self.fpc, self.avg_dim)
        x = self.gaze_fc(x)
        if self.save_gaze_out:
            self.gaze_out = x[:, :, :self.emb_dim-self.leaky]
        
        q, k, v = self.MHA_Q_fc(x), self.MHA_K_fc(x), self.MHA_V_fc(x)
        x, _ = self.multihead_attn(q, k, v)

        x = x.reshape(batch_size * self.fpc, self.MHA_dim)
        x = self.MHA_comp(x)
        x = x.reshape(batch_size, self.fpc * self.MHA_comp_dim)
        x = self.last_fc(x)
        x[:, :2] = self.softmax(x[:, :2])

        return x