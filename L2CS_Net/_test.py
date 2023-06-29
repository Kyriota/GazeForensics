import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

import _datasets as datasets
from utils import select_device, natural_keys, gazeto3d, angular
from _backend import L2CS


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze estimation using L2CSNet .')
     # Gaze360
    parser.add_argument(
        '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
        default='datasets/Gaze360/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
        default='datasets/Gaze360/Label/test.label', type=str)
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
        '--dataset', dest='dataset', help='gaze360, mpiigaze',
        default= "gaze360", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path to the folder contains models.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4-lr', type=str)
    parser.add_argument(
        '--evalpath', dest='evalpath', help='path for the output evaluating gaze test.',
        default="evaluation/L2CS-gaze360-_loader-180-4-lr", type=str)
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=100, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    batch_size=args.batch_size
    arch=args.arch
    data_set=args.dataset
    evalpath =args.evalpath
    snapshot_path = args.snapshot
    bins=90
    angle=None
    bin_width=None

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
            # transforms.Resize(32),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
    ]

    criterion = nn.MSELoss().cuda(gpu)
    
    if data_set=="gaze360":
        
        gaze_dataset=datasets.Gaze360(args.gaze360label_dir,args.gaze360image_dir, transformations, 180, 4, train=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=gaze_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

        

        if not os.path.exists(evalpath):
            os.makedirs(evalpath)


        # list all epochs for testing
        folder = os.listdir(snapshot_path)
        folder.sort(key=natural_keys)
        softmax = nn.Softmax(dim=1)
        with open(os.path.join(evalpath,data_set+".log"), 'w') as outfile:
            configuration = f"\ntest configuration = gpu_id={gpu}, batch_size={batch_size}, model_arch={arch}\nStart testing dataset={data_set}----------------------------------------\n"
            print(configuration)
            outfile.write(configuration)
            epoch_list=[]
            avg_yaw=[]
            avg_pitch=[]
            avg_MSE=[]

            teacher_model = L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90, eval_mode=False, need_decoder=False)
            saved_state_dict = torch.load('models/L2CSNet_gaze360.pkl')
            print(teacher_model.load_state_dict(saved_state_dict, strict=False))
            teacher_model.cuda(gpu)
            teacher_model.eval()

            for epochs in folder:

                model = L2CS(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 90, eval_mode=False, need_decoder=True)
                saved_state_dict = torch.load(os.path.join(snapshot_path, epochs))
                print(model.load_state_dict(saved_state_dict, strict=False))
                
                model.cuda(gpu)
                model.eval()
                total = len(test_loader.dataset)
                idx_tensor = [idx for idx in range(90)]
                idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
                loss_sum = 0
                
                
                with torch.no_grad():           
                    for _, ((images_ori, images_comp), labels_gaze, cont_labels_gaze,name) in enumerate(test_loader):

                        images_ori = Variable(images_ori).cuda(gpu)
                        images_comp = Variable(images_comp).cuda(gpu)

                        model_out = model(images_comp)
                        teacher_out = teacher_model(images_ori)

                        # MSE loss
                        loss = criterion(model_out, teacher_out)
                        
                        loss_sum += loss.item()*len(labels_gaze)
                            
                    x = ''.join(filter(lambda i: i.isdigit(), epochs))
                    epoch_list.append(x)
                    avg_MSE.append(loss_sum/total)
                    loger = f"[{epochs}---{args.dataset}] Total Num:{total},MSE:{loss_sum/total}\n"
                    outfile.write(loger)
                    print(loger)
        
        fig = plt.figure(figsize=(14, 8))        
        plt.xlabel('epoch')
        plt.ylabel('avg')
        plt.title('Gaze angular error')
        plt.legend()
        plt.plot(epoch_list, avg_MSE, color='k', label='mae')
        fig.savefig(os.path.join(evalpath,data_set+".png"), format='png')
        plt.show()
            
           
