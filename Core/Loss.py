from torch import nn
import torch



def get_bonus(output, target):
    '''
    NOT WORKING PROPERLY YET
    '''
    logits = output[:, :2]
    label = target[:, 1:]
    requested_bonus = output[:, 2:]
    extent = torch.zeros_like(requested_bonus)

    extent[label == 1] = (logits[:, 1:][label == 1] - 0.5) * 2
    extent[label == 0] = (logits[:, :1][label == 0] - 0.5) * 2

    bonus = requested_bonus * extent

    return bonus



class CustomLoss:
    def __init__(self, params_loss):
        self.FN_w = params_loss['FN_w'] # False negative weight
        self.FN_bound = params_loss['FN_bound'] # False negative boundary
        self.BCELoss = nn.BCELoss(reduction='none').cuda()
        self.bonus_weight = params_loss['bonus_weight']
        self.out_loss = None
        self.gaze_loss = None
    
    
    def __call__(self, output, target):
        output_real = output['out'][:, 1:2]
        false_negative = torch.logical_and(target['out'][:, 1:] == 1, output_real < self.FN_bound).squeeze()
        self.out_loss = self.BCELoss(output['out'][:, :2], target['out']).mean(dim=1)
        self.out_loss[false_negative] *= 1 + (self.FN_bound - output_real[false_negative].squeeze()) / self.FN_bound * self.FN_w

        if self.bonus_weight > 0:
            bonus = get_bonus(output['out'][:, :2], target['out'])
            self.out_loss -= bonus * self.bonus_weight
        
        self.out_loss = self.out_loss.mean()
        return self.out_loss



class TrainLoss:
    def __init__(self, params_loss):
        self.loss_func = None
        self.gaze_weight = params_loss['gaze_weight']
        self.bonus_weight = params_loss['bonus_weight']
        self.out_loss = None
        self.gaze_loss = None
        if self.gaze_weight > 0.0:
            self.loss_func = [nn.BCELoss().cuda(), nn.MSELoss().cuda()]
        else:
            self.loss_func = nn.BCELoss().cuda()


    def __call__(self, output, target):
        loss = 0

        if self.gaze_weight > 0.0:
            self.gaze_loss = self.loss_func[1](output['gaze'], target['gaze'])
            self.out_loss = self.loss_func[0](output['out'][:, :2], target['out'])
            loss = self.gaze_weight * self.gaze_loss + self.out_loss
        else:
            self.out_loss = self.loss_func(output['out'][:, :2], target['out'])
            loss = self.out_loss
        
        if self.bonus_weight > 0:
            bonus = get_bonus(output['out'], target['out'])
            loss -= bonus.mean() * self.bonus_weight

        return loss



class TestLoss:
    def __init__(self):
        self.loss_func = nn.BCELoss().cuda()

    def __call__(self, output, target):
        self.out_loss = self.loss_func(output[:, :2], target)
        return self.out_loss