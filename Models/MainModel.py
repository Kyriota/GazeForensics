import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import BasicBlock

from Models.Resnet import resnet18
from Models.FcBlock import FC_block
from Models.GazeBackend import GazeBackend
from Config import Config, load_lagacy_config
from ConfigTypes import ModelMidSizes


def InitModel(config: Config):
    if type(config.model) is dict:
        config = load_lagacy_config(config)

    torch.manual_seed(config.model.seed)

    gaze_backend = None
    if config.model.freeze_backend:
        gaze_backend = GazeBackend(
            BasicBlock,
            [2, 2, 2, 2],
            emb_dim=config.model.emb_dim,
        )
        state_dict = torch.load(config.model.gaze_backend_path)
        print('Loading gaze backend for frozen backend')
        print(gaze_backend.load_state_dict(state_dict, strict=False))
        gaze_backend = nn.DataParallel(gaze_backend).cuda()

    model = GazeForensics(
        save_gaze_out=config.loss.gaze_weight > 0.0,
        leaky_dim=config.model.leaky,
        gaze_emb_dim=config.model.emb_dim,
        head_num=config.model.head_num,
        dim_per_head=config.model.dim_per_head,
        comp_dim=config.model.comp_dim,
        mid_sizes=config.model.mid_sizes,
        gaze_backend=gaze_backend,
    )

    if not config.model.freeze_backend:
        # load gaze backend
        state_dict = torch.load(config.model.gaze_backend_path)
        print('Loading gaze backend for finetuning')
        print(model.base_model.load_state_dict(state_dict, strict=False))

    model = nn.DataParallel(model).cuda()

    if config.model.freeze_backend:
        for param in model.module.gaze_backend.parameters():
            param.requires_grad = False

    return model


class GazeForensics(nn.Module):
    def __init__(
        self,
        save_gaze_out=False,
        leaky_dim=0,
        gaze_emb_dim=512,
        head_num=4,
        dim_per_head=32,
        comp_dim=64,
        mid_sizes={
            ModelMidSizes.gaze_fc: None,
            ModelMidSizes.MHA_fc: None,
            ModelMidSizes.MHA_comp: None,
            ModelMidSizes.last_fc: None,
        },
        gaze_backend=None,  # If gaze backend is given, it will be frozen
    ):
        nn.Module.__init__(self)

        self.save_gaze_out = save_gaze_out
        self.freeze_backend = gaze_backend is not None
        if self.freeze_backend:
            assert type(gaze_backend) is nn.DataParallel and type(
                gaze_backend.module) is GazeBackend
            self.gaze_backend = gaze_backend
            self.save_gaze_out = False

        self.leaky_dim = leaky_dim
        self.leaky_dim = leaky_dim
        self.avg_dim = 512
        self.gaze_emb_dim = gaze_emb_dim
        self.total_emb_dim = gaze_emb_dim + leaky_dim
        self.gaze_emb_dim = gaze_emb_dim
        self.total_emb_dim = gaze_emb_dim + leaky_dim
        self.MHA_dim = head_num * dim_per_head
        self.MHA_comp_dim = comp_dim
        self.fpc = 14  # frames per clip
        self.gaze_out = None
        self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()
        # self.sigmoid = nn.Sigmoid()
        last_fc_input_size = self.MHA_comp_dim * self.fpc

        # Set default values for mid_sizes
        if mid_sizes[ModelMidSizes.gaze_fc] is None:
            mid_sizes[ModelMidSizes.gaze_fc] = [self.total_emb_dim]
        if mid_sizes[ModelMidSizes.MHA_fc] is None:
            mid_sizes[ModelMidSizes.MHA_fc] = []
        if mid_sizes[ModelMidSizes.MHA_comp] is None:
            mid_sizes[ModelMidSizes.MHA_comp] = [128]
        if mid_sizes[ModelMidSizes.last_fc] is None:
            mid_sizes[ModelMidSizes.last_fc] = [
                last_fc_input_size // 2,
                last_fc_input_size // 4,
            ]

        if not self.freeze_backend or self.leaky_dim > 0:
            print('Initializing base model')
            self.base_model = resnet18(pretrained=True)
            if not self.freeze_backend:
                self.gaze_fc = FC_block(
                    self.avg_dim, self.total_emb_dim, mid_sizes=mid_sizes[ModelMidSizes.gaze_fc])
            else:  # frozen backend, but leaky > 0
                self.gaze_fc = FC_block(
                    self.avg_dim, self.leaky_dim, mid_sizes=mid_sizes[ModelMidSizes.gaze_fc])

        self.MHA_Q_fc = FC_block(
            self.total_emb_dim, self.MHA_dim, mid_sizes=mid_sizes[ModelMidSizes.MHA_fc])
        self.MHA_K_fc = FC_block(
            self.total_emb_dim, self.MHA_dim, mid_sizes=mid_sizes[ModelMidSizes.MHA_fc])
        self.MHA_V_fc = FC_block(
            self.total_emb_dim, self.MHA_dim, mid_sizes=mid_sizes[ModelMidSizes.MHA_fc])
        self.multihead_attn = nn.MultiheadAttention(
            self.MHA_dim, head_num, batch_first=True)
        self.MHA_comp = FC_block(
            self.MHA_dim, self.MHA_comp_dim, mid_sizes=mid_sizes[ModelMidSizes.MHA_comp])
        self.last_fc = FC_block(last_fc_input_size, 2,
                                mid_sizes=mid_sizes[ModelMidSizes.last_fc])

        # # ----------------- Debug -----------------
        # self.debug_record = {
        #     'base_out': [],
        #     'q_out': [],
        #     'k_out': [],
        #     'v_out': [],
        #     'attn_out': [],
        #     'MHA_comp_out': [],
        #     'last_fc_out': [],
        # }

    def forward(self, x):
        batch_size = x.size(0)
        if self.freeze_backend:
            frozen_out = self.gaze_backend(
                x.view(-1, 3, 224, 224)).view(batch_size, self.fpc, self.gaze_emb_dim)
        if not self.freeze_backend or self.leaky_dim > 0:
            x = self.base_model(
                x.view((batch_size * self.fpc, 3) + x.size()[-2:]))
            x = x.view(batch_size, self.fpc, self.avg_dim)
            x = self.gaze_fc(x)
        if self.save_gaze_out:
            self.gaze_out = x[:, :, :self.total_emb_dim-self.leaky_dim]

        if self.freeze_backend:
            x = frozen_out if self.leaky_dim == 0 else torch.cat(
                (x, frozen_out), dim=2)

        q, k, v = self.MHA_Q_fc(x), self.MHA_K_fc(x), self.MHA_V_fc(x)
        x, _ = self.multihead_attn(q, k, v)

        x = x.reshape(batch_size * self.fpc, self.MHA_dim)
        x = self.MHA_comp(x)
        x = x.reshape(batch_size, self.fpc * self.MHA_comp_dim)
        x = self.last_fc(x)
        x = self.softmax(x)

        return x

    # # ----------------- Debug -----------------
    # def forward(self, x):
    #     batch_size = x.size(0)
    #     if self.freeze_backend:
    #         frozen_out = self.gaze_backend(
    #             x.view(-1, 3, 224, 224)).view(batch_size, self.fpc, self.gaze_emb_dim)
    #     if not self.freeze_backend or self.leaky_dim > 0:
    #         x = self.base_model(
    #             x.view((batch_size * self.fpc, 3) + x.size()[-2:]))
    #         x = x.view(batch_size, self.fpc, self.avg_dim)
    #         x = self.gaze_fc(x)
    #     if self.save_gaze_out:
    #         self.gaze_out = x[:, :, :self.total_emb_dim-self.leaky_dim]
    #     self.debug_record['base_out'].append(x.detach().cpu().numpy())

    #     if self.freeze_backend:
    #         x = frozen_out if self.leaky_dim == 0 else torch.cat(
    #             (x, frozen_out), dim=2)

    #     q, k, v = self.MHA_Q_fc(x), self.MHA_K_fc(x), self.MHA_V_fc(x)
    #     self.debug_record['q_out'].append(q.detach().cpu().numpy())
    #     self.debug_record['k_out'].append(k.detach().cpu().numpy())
    #     self.debug_record['v_out'].append(v.detach().cpu().numpy())
    #     x, _ = self.multihead_attn(q, k, v)
    #     self.debug_record['attn_out'].append(x.detach().cpu().numpy())

    #     x = x.reshape(batch_size * self.fpc, self.MHA_dim)
    #     x = self.MHA_comp(x)
    #     self.debug_record['MHA_comp_out'].append(x.detach().cpu().numpy())
    #     x = x.reshape(batch_size, self.fpc * self.MHA_comp_dim)
    #     x = self.last_fc(x)
    #     self.debug_record['last_fc_out'].append(x.detach().cpu().numpy())
    #     x = self.softmax(x)

    #     return x
