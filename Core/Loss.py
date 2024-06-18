from torch import nn
import torch
from Config import ParamLoss


def smooth_label(label, alpha, beta):
    rand = torch.rand(label.shape[0], 1) * beta + alpha
    rand = torch.cat([rand, rand], dim=1).cuda(non_blocking=True)
    label = torch.abs(label - rand)
    return label


class TrainLoss:
    def __init__(self, params_loss: ParamLoss):
        self.loss_func = None
        self.gaze_weight = params_loss.gaze_weight
        self.gaze_gamma = params_loss.gaze_gamma
        # self.bonus_weight = params_loss['bonus_weight']
        self.out_loss = None
        self.gaze_loss = None
        if self.gaze_weight > 0.0:
            self.loss_func = [nn.BCELoss().cuda(), nn.MSELoss().cuda()]
        else:
            self.loss_func = nn.BCELoss().cuda()

    def gaze_weight_schedule(self):
        self.gaze_weight *= self.gaze_gamma
        print('Adjusting gaze weight to', self.gaze_weight)

    def __call__(self, output, target):
        loss = 0

        if self.gaze_weight > 0.0:
            self.gaze_loss = self.loss_func[1](output['gaze'], target['gaze'])
            self.out_loss = self.loss_func[0](
                output['out'][:, :2], target['out'])
            loss = self.gaze_weight * self.gaze_loss + self.out_loss
        else:
            self.out_loss = self.loss_func(output['out'][:, :2], target['out'])
            loss = self.out_loss

        # if self.bonus_weight > 0:
        #     bonus = get_bonus(output['out'], target['out'])
        #     self.bonus_loss = bonus.mean()
        #     loss -= self.bonus_loss * self.bonus_weight

        return loss


class TestLoss:
    def __init__(self):
        self.loss_func = nn.BCELoss().cuda()

    def __call__(self, output, target):
        self.out_loss = self.loss_func(output[:, :2], target)
        return self.out_loss


# def get_bonus(output, target):
#     logits = output[:, :2]
#     label = target[:, 1:]
#     requested_bonus = output[:, 2:]
#     extent = torch.zeros_like(requested_bonus)

#     extent[label == 1] = (logits[:, 1:][label == 1] - 0.5) * 2
#     extent[label == 0] = (logits[:, :1][label == 0] - 0.5) * 2

#     bonus = requested_bonus * extent

#     return bonus


# class CustomLoss:
#     def __init__(self, params_loss):
#         self.FN_w = params_loss['FN_w'] # False negative weight
#         self.FN_bound = params_loss['FN_bound'] # False negative boundary
#         self.BCELoss = nn.BCELoss(reduction='none').cuda()
#         self.bonus_weight = params_loss['bonus_weight']
#         self.out_loss = None
#         self.gaze_loss = None


#     def __call__(self, output, target):

#         self.out_loss = self.BCELoss(output['out'][:, :2], target['out']).mean(dim=1)

#         if self.FN_w > 0:
#             output_real = output['out'][:, 1:2]
#             false_negative = torch.logical_and(target['out'][:, 1:] == 1, output_real < self.FN_bound).squeeze()
#             self.out_loss[false_negative] *= 1 + (self.FN_bound - output_real[false_negative].squeeze()) / self.FN_bound * self.FN_w

#         if self.bonus_weight > 0:
#             bonus = get_bonus(output['out'][:, :2], target['out'])
#             self.out_loss -= bonus * self.bonus_weight

#         self.out_loss = self.out_loss.mean()
#         return self.out_loss
