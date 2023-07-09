import torch
import torch.nn as nn

from Models.Resnet import resnet18
from Config import Config



def InitModel(config:Config):
    torch.manual_seed(config.model['seed'])
    model = GazeForensics(
        save_emb_out=config.loss['gaze_weight'] > 0,
        leaky=config.model['leaky'],
        emb_dim=config.model['emb_dim'],
        head_num=config.model['head_num'],
        dim_per_head=config.model['dim_per_head'],
        comp_dim=config.model['comp_dim'],
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



class ImageDecoder(nn.Module):
    # This class decode the image from a vector using a series of deconvolution layers
    def __init__(self):
        nn.Module.__init__(self)
        self.emb_dim = 512
        self.fc = FC_block(self.emb_dim, 7*7*256, mid_sizes=[512, 1024, 1024])
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        x = self.deconv(x)
        return x



class GazeForensics(nn.Module):
    def __init__(
            self,
            save_emb_out=False,
            leaky=0,
            emb_dim=512,
            head_num=4,
            dim_per_head=32,
            comp_dim=64,
        ):
        nn.Module.__init__(self)

        self.save_emb_out = save_emb_out

        self.leaky = leaky
        self.avg_dim = 512
        self.total_emb_dim = emb_dim + leaky
        self.MHA_dim = head_num * dim_per_head
        self.MHA_comp_dim = comp_dim
        self.fpc = 14 # frames per clip
        self.emb_out = None
        self.softmax = nn.Softmax(dim=1)

        self.base_model = resnet18(pretrained=True)
        self.multihead_attn = nn.MultiheadAttention(self.MHA_dim, head_num, batch_first=True)
        self.MHA_comp = FC_block(self.MHA_dim, self.MHA_comp_dim, mid_sizes=[
            256,
            self.MHA_comp_dim
        ])
        last_fc_input_size = self.MHA_comp_dim * self.fpc
        self.last_fc = FC_block(last_fc_input_size, 3, mid_sizes=[
            last_fc_input_size // 2,
            last_fc_input_size // 4,
        ])

        self.gaze_fc = FC_block(self.avg_dim, self.total_emb_dim, mid_sizes=[self.total_emb_dim])
        self.MHA_Q_fc = FC_block(self.total_emb_dim, self.MHA_dim, mid_sizes=[self.MHA_dim, self.MHA_dim])
        self.MHA_K_fc = FC_block(self.total_emb_dim, self.MHA_dim, mid_sizes=[self.MHA_dim, self.MHA_dim])
        self.MHA_V_fc = FC_block(self.total_emb_dim, self.MHA_dim, mid_sizes=[self.MHA_dim, self.MHA_dim])

        # To minimize the influence on other layers within a fixed seed,
        # initialize layers influenced by hyperparameters after other layers have been initialized.



    def forward(self, x):
        batch_size = x.size(0)
        x = self.base_model(x.view((batch_size * self.fpc, 3) + x.size()[-2:]))
        x = x.view(batch_size, self.fpc, self.avg_dim)
        x = self.gaze_fc(x)
        if self.save_emb_out:
            self.emb_out = x[:, :, :self.total_emb_dim-self.leaky]
        
        q, k, v = self.MHA_Q_fc(x), self.MHA_K_fc(x), self.MHA_V_fc(x)
        x, _ = self.multihead_attn(q, k, v)

        x = x.reshape(batch_size * self.fpc, self.MHA_dim)
        x = self.MHA_comp(x)
        x = x.reshape(batch_size, self.fpc * self.MHA_comp_dim)
        x = self.last_fc(x)
        x[:, :2] = self.softmax(x[:, :2])

        return x