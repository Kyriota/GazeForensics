import os
import argparse
import time

import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import _datasets as datasets
from _backend import L2CS
from utils import select_device
import random
from PIL import Image
import numpy as np
from datetime import datetime


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Gaze estimation using L2CSNet.')
    # Gaze360
    parser.add_argument(
        '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
        default='datasets/Gaze360/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
        default='datasets/Gaze360/Label/train.label', type=str)
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/Label', type=str)

    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='mpiigaze, rtgene, gaze360, ethgaze',
        default= "gaze360", type=str)
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='snapshots/', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=60, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=1, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    parser.add_argument(
        '--alpha', dest='alpha', help='Regression loss coefficient.',
        default=1, type=float)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00001, type=float)
    parser.add_argument(
        '--emb_dim', dest='emb_dim', help='Dimension of the embedding vector.',
        default=128, type=int)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args

def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw_gaze, model.fc_pitch_gaze]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param
                
def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict, strict=False)

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    data_set=args.dataset
    alpha = args.alpha
    output=args.output
    emb_dim = args.emb_dim

    class RandomResizeTransforms(object):

        def __call__(self, img):
            if random.random() < 0.5:
                size = random.randint(32, 223)
                return img.resize((size, size))
                # return img.resize((32, 32))
            else:
                return img

    transformations = [
        transforms.Compose([
            transforms.Resize(448),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
        transforms.Compose([
            RandomResizeTransforms(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
    ]
    
    
    if data_set=="gaze360":
        model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 90, eval_mode=False, emb_dim=emb_dim)
        load_filtered_state_dict(model, model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth'))
        teacher_model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90, eval_mode=False, emb_dim=0)
        saved_state_dict = torch.load('models/L2CSNet_gaze360.pkl')
        print(teacher_model.load_state_dict(saved_state_dict, strict=False))
        if args.snapshot:
            saved_state_dict = torch.load(args.snapshot)
            print(model.load_state_dict(saved_state_dict, strict=False))
        
        model.cuda(gpu)
        teacher_model.cuda(gpu)
        dataset=datasets.Gaze360(args.gaze360label_dir, args.gaze360image_dir, transformations, 180, 4)
        print('Loading data.')
        train_loader_gaze = DataLoader(
            dataset=dataset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=4,
            pin_memory=True)
        torch.backends.cudnn.benchmark = True

        summary_name = '{}_{}'.format('L2CS-gaze360-', datetime.now().strftime("%y%m%d%H%M%S"))
        output=os.path.join(output, summary_name)
        if not os.path.exists(output):
            os.makedirs(output)

        criterion = nn.MSELoss().cuda(gpu)

        # Optimizer gaze
        optimizer_gaze = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_gaze, mode='min', factor=0.5, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

        configuration = f"\ntrain configuration, gpu_id={args.gpu_id}, batch_size={batch_size}, model_arch={args.arch}\nStart testing dataset={data_set}, loader={len(train_loader_gaze)}------------------------- \n"
        print(configuration)

        model.train()
        teacher_model.eval()
        
        for epoch in range(num_epochs):

            sum_l2_loss = sum_MSE_loss = cnt = 0

            for i, ((images_ori, images_comp), labels_gaze, cont_labels_gaze,name) in enumerate(train_loader_gaze):

                images_ori = Variable(images_ori).cuda(gpu)
                images_comp = Variable(images_comp).cuda(gpu)

                model_out, code = model(images_comp)
                with torch.no_grad():
                    teacher_out = teacher_model(images_ori)

                # MSE loss + L2 loss
                l2_loss = torch.norm(code, p=2, dim=1).mean()
                MSE_loss = criterion(model_out, teacher_out)
                alpha = 0.25
                loss = MSE_loss + l2_loss * alpha

                sum_l2_loss += float(l2_loss.cpu().detach())
                sum_MSE_loss += float(MSE_loss.cpu().detach())
                cnt += 1

                optimizer_gaze.zero_grad()
                loss.backward()
                optimizer_gaze.step()

                if (i+1) % 100 == 0:
                    print('Epoch [%d/%d], Iter [%d/%d] Losses: '
                          'MSE %.4f, L2 %.4f' % (
                                epoch + 1,
                                num_epochs,
                                i + 1,
                                len(dataset) // batch_size,
                                MSE_loss,
                                l2_loss
                            )
                        )
                    sum_l2_loss = sum_MSE_loss = cnt = 0
                    
            # scheduler.step()
          
            if epoch % 1 == 0 and epoch < num_epochs:
                print('Taking snapshot...',
                    torch.save(model.state_dict(),
                                output +'/'+
                                '_epoch_' + str(epoch+1) + '.pkl')
                    )