import sys

if 'Models/' not in sys.path:
    sys.path.append('Models/')
from model import GazeForensics
from L2CS_Net import L2CS

if 'Utils/' not in sys.path:
    sys.path.append('Utils/')
from FileOperation import *
from DatasetClipping import ClipTrainDatasetParallel
from Visualization import ProgressBar, PlotHistory

from Config import basic_param, prep_param, model_param, opti_param, train_param

import torch
import torch.nn as nn
import torch.utils.data as TDM # torch data module
from torchvision.models import resnet
# from torchsummary import summary

import cv2
from PIL import Image
import threading
import time
import numpy as np
import math



class DatasetLoader(TDM.Dataset):
    def __init__(self):
        clippedDataPath = prep_param['tempDir'] + 'current/'
        self.transform = prep_param['transform']
        self.dataset = [clippedDataPath + i for i in fileWalk(clippedDataPath)]
        self.dataset = [[0 if 'fake' in path else 1, path] for path in self.dataset]


    def __getitem__(self, index):
        sign, vid_path = self.dataset[index]
        inputTensor = torch.FloatTensor(14, 3, 224, 224)
        vidcap = cv2.VideoCapture(vid_path)
        assert vidcap.isOpened(), '\n>>> ERROR: Failed to open the video: ' + vid_path + '\n'
        assert vidcap.get(cv2.CAP_PROP_FRAME_COUNT) == 14, '\n>>> ERROR: The video is not 14 frames long: ' + vid_path + '\n'
        for i in range(14):
            _, image = vidcap.read()
            inputTensor[i,...] = self.transform(Image.fromarray(image))
        vidcap.release()
        inputTensor = inputTensor[:, [2, 1, 0], :, :]

        return [sign, inputTensor]
    

    def __len__(self):
        return len(self.dataset)
    


class Loss:
    def __init__(self):
        self.loss_func = None
        self.gaze_loss_weight = opti_param['gaze_loss_weight']
        if self.gaze_loss_weight > 0.0:
            self.loss_func = [nn.BCELoss().cuda(), nn.MSELoss().cuda()]
        else:
            self.loss_func = nn.BCELoss().cuda()


    def __call__(self, output, target):
        if self.gaze_loss_weight > 0.0:
            self.gaze_loss = self.loss_func[1](output['gaze'], target['gaze'])
            self.out_loss = self.loss_func[0](output['out'], target['out'])
            return self.gaze_loss_weight * self.gaze_loss + self.out_loss
        else:
            self.out_loss = self.loss_func(output['out'], target['out'])
            return self.out_loss
    


class Optimizer:
    def __init__(self, model):
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opti_param['lr'],
            weight_decay=opti_param['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=opti_param['step_size'],
            gamma=opti_param['gamma'],
            last_epoch=-1,
            verbose=True
        )
    

    def zero_grad(self):
        self.optimizer.zero_grad()

    
    def optimizer_step(self):
        self.optimizer.step()


    def scheduler_step(self):
        self.scheduler.step()
        


class TrainManager:
    def __init__(self):

        def InitModel():
            model = GazeForensics(
                leaky=model_param['leaky'],
                save_backend_output=self.use_gaze,
                arch=model_param['arch']
            )
            model = nn.DataParallel(model).cuda()
            return model

        self.gaze_backend = None
        self.dataset = None
        self.data_loader = None
        self.output_lock = threading.Lock()
        self.use_gaze = opti_param['gaze_loss_weight'] > 0.0
        self.history = {
            'out_loss': [],
            'acc': [],
            'gaze_loss': [],
            'total_loss': [],
        }

        # Init the model
        if model_param['model_path'] is None:
            self.model = InitModel()
        else:
            raise NotImplementedError('Loading model is not implemented yet.')
            # Loading model needs to read other info like optimizer status, epoch num, etc.
            # This will be implemented later.

        # Init optimizer, scheduler and loss function
        self.optimizer = Optimizer(self.model)
        self.loss = Loss()
    
    
    def train(self):

        def get_preprocess_thread(epoch):
            return threading.Thread(
                target=ClipTrainDatasetParallel,
                kwargs={
                    'json_path': prep_param['trainCateDictDir'],
                    'output_path': prep_param['tempDir'] + 'next/',
                    'output_lock': self.output_lock,
                    'thread_num': prep_param['thread_num'],
                    'utilization_percentage': prep_param['utilization_percentage'],
                    'epoch': epoch,
                }
            )


        def substitute_dataset():
            rm(prep_param['tempDir'] + 'current/', r=True)
            mv(prep_param['tempDir'] + 'next/', prep_param['tempDir'] + 'current/')
        

        def get_preprocess_condition(epoch):
            if epoch == train_param['num_epochs'] - 1:
                return False
            if prep_param['prep_every_epoch'] is None:
                return prep_param['utilization_percentage'] == 1
            return prep_param['prep_every_epoch']
        
        
        def init_data_loader():
            self.dataset = DatasetLoader()
            self.data_loader = TDM.DataLoader(
                self.dataset,
                batch_size=train_param['batch_size'],
                shuffle=True,
                num_workers=train_param['num_workers'],
                pin_memory=True
            )


        def train(epoch):

            def get_acc(output, sign):
                acc = torch.eq((output>0.5).int(), sign).tolist()
                acc = [i[0] and i[1] for i in acc]
                return np.mean(acc)


            def save_to_history(temp_history, out_loss, acc, total_loss=None, gaze_loss=None):
                temp_history['out_loss'].append(out_loss)
                temp_history['acc'].append(acc)
                if self.use_gaze:
                    temp_history['total_loss'].append(total_loss)
                    temp_history['gaze_loss'].append(gaze_loss)
            

            def extend_history(temp_history):
                self.history['out_loss'].extend(temp_history['out_loss'])
                self.history['acc'].extend(temp_history['acc'])
                if self.use_gaze:
                    self.history['total_loss'].extend(temp_history['total_loss'])
                    self.history['gaze_loss'].extend(temp_history['gaze_loss'])
            
            
            self.output_lock.acquire()
            print('-' * 50)
            progress_bar = ProgressBar(
                'green',
                'Epoch {}/{}'.format(epoch+1, train_param['num_epochs']),
                len(self.data_loader),
            )
            print('-' * 50)
            self.output_lock.release()

            temp_history = {
                'out_loss': [],
                'acc': [],
                'total_loss': [],
                'gaze_loss': [],
            }

            self.model.train()
            torch.manual_seed(epoch)

            for sign, frames in self.data_loader:
                sign = nn.functional.one_hot(sign.cuda(non_blocking=True), num_classes=2).float()
                sign_var = torch.autograd.Variable(sign).cuda()
                frames_var = torch.autograd.Variable(frames.cuda(non_blocking=True)).cuda()

                output = self.model(frames_var)

                if self.use_gaze:
                    gazeTarget = self.gaze_backend(frames_var.view(-1, 3, 224, 224))
                    gazeEmb = self.model.module.backend_out.view(-1, gazeTarget.shape[1])

                output = {
                    'out': output,
                    'gaze': gazeEmb if self.use_gaze else None
                }
                target = {
                    'out': sign_var,
                    'gaze': gazeTarget if self.use_gaze else None
                }

                total_loss = self.loss(output, target)
                acc = get_acc(output['out'], sign)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.optimizer_step()

                save_to_history(
                    temp_history,
                    float(self.loss.out_loss.cpu().detach()),
                    float(acc),
                    float(total_loss.cpu().detach()) if self.use_gaze else None,
                    float(self.loss.gaze_loss.cpu().detach()) if self.use_gaze else None
                )

                progress_bar.Update(
                    '- Mean Out Loss: {:.4f}<br>- Mean Acc: {:.4f}'.format(
                        np.mean(temp_history['out_loss']),
                        np.mean(temp_history['acc']),
                    ) + ('<br>- Mean Total Loss: {:.4f}<br>- Mean Gaze Loss: {:.4f}'.format(
                        np.mean(temp_history['total_loss']),
                        np.mean(temp_history['gaze_loss']),
                    ) if self.use_gaze else '')
                )

            extend_history(temp_history)
        

        def save_checkpoint(epoch):
            checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.optimizer.state_dict(),
                'scheduler': self.optimizer.scheduler.state_dict(),
                'history': self.history,
            }
            torch.save(checkpoint, basic_param['checkpointDir'] + basic_param['tryID'] + '/checkpoint_{}.pth'.format(epoch+1))


        # Check if tryID exists in checkpointDir
        if basic_param['tryID'] is None:
            print('>> No tryID specified. This run will not be saved.')
        elif basic_param['tryID'] == 'temp':
            print('>> tryID is "temp". Last run in temp will be overwritten.')
            rm(basic_param['checkpointDir'] + 'temp/', r=True)
            mkdir(basic_param['checkpointDir'] + 'temp/')
        elif basic_param['tryID'] in ls(basic_param['checkpointDir']):
            raise Exception('tryID already exists in checkpointDir, please change tryID or delete the existing one')
        else:
            mkdir(basic_param['checkpointDir'] + basic_param['tryID'] + '/')
        
        # Preprocess the training dataset
        preprocess_thread = get_preprocess_thread(0)
        preprocess_thread.start()

        if self.use_gaze:
            # Init gaze model
            self.gaze_backend = L2CS(resnet.Bottleneck, [3, 4, 6, 3])
            state_dict = torch.load('Checkpoints/L2CSNet_gaze360.pkl')
            self.gaze_backend.load_state_dict(state_dict, strict=False)
            self.gaze_backend = nn.DataParallel(self.gaze_backend).cuda()
            self.gaze_backend.train() # Set to train mode to enable batch normalization

        # Wait for the first preprocessing to finish
        preprocess_thread.join()
        
        # substitute_dataset()
        init_data_loader()

        # Start training
        for epoch in range(train_param['num_epochs']):

            if get_preprocess_condition(epoch):
                # Preprocess the training dataset
                preprocess_thread = get_preprocess_thread(epoch+1)
                preprocess_thread.start()

            train(epoch)
            PlotHistory(
                self.history,
                self.use_gaze,
                smooth_window_size=math.ceil(512/train_param['batch_size'])
            )
            save_checkpoint(epoch)
            self.optimizer.scheduler_step()

            if get_preprocess_condition(epoch):
                # Wait for the preprocessing to finish
                preprocess_thread.join()

                substitute_dataset()
                init_data_loader()