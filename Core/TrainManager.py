from Models.MainModel import InitModel
from Models.Resnet import resnet18
from Utils.FileOperation import mkdir, rm, mv, fileExist, fileWalk
from Utils.Visualization import ProgressBar, PlotHistory
from Preprocess.DatasetClipping import ClipTrainDatasetParallel
from Core.DatasetLoader import DatasetLoader
from Core.Loss import TrainLoss, CustomLoss
from Config import Config

import torch
from torch import nn
import torch.utils.data as TDM # torch data module

import threading
from datetime import datetime
import numpy as np
import math
import json



class TrainManager:

    def __init__(self, config:Config):

        self.config = config
        self.gaze_backend = None
        self.data_loader = None
        self.output_lock = threading.Lock()
        self.use_gaze = self.config.loss['gaze_weight'] > 0.0
        self.temp_history = None
        self.history = {
            'out_loss': [],
            'acc': [],
            'gaze_loss': [],
            'total_loss': [],
            'len_per_epoch': None,
        }

        # Init the model
        self.model = InitModel(config)

        # Init optimizer, scheduler and loss function
        self.optimizer = self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.opti['lr'],
            weight_decay=config.opti['weight_decay']
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.opti['step_size'],
            gamma=config.opti['gamma'],
            last_epoch=-1,
            verbose=True
        )
        if config.loss['loss_func'] == 'standard':
            self.loss = TrainLoss(config.loss)
        elif config.loss['loss_func'] == 'custom':
            self.loss = CustomLoss(config.loss)

        # Load checkpoint
        if self.config.model['checkpoint']:
            checkpoint = torch.load(self.config.model['checkpoint'])
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        if self.use_gaze:
            # # Init gaze backend
            self.gaze_backend = resnet18(pretrained=False)
            state_dict = torch.load(self.config.basic['checkpointDir'] + 'gazeBackend.pkl')
            self.gaze_backend.load_state_dict(state_dict, strict=False)
            self.gaze_backend = nn.DataParallel(self.gaze_backend).cuda()
            self.gaze_backend.eval()
    
    
    def train(self):

        def get_preprocess_thread(epoch):
            return threading.Thread(
                target=ClipTrainDatasetParallel,
                kwargs={
                    'json_path': self.config.prep['trainCateDictDir'],
                    'output_path': self.config.prep['tempDir'] + 'next/',
                    'output_lock': self.output_lock,
                    'thread_num': self.config.prep['thread_num'],
                    'util_percent': self.config.prep['util_percent'],
                    'epoch': epoch,
                }
            )


        def update_dataset():
            rm(self.config.prep['tempDir'] + 'current/', r=True)
            mv(self.config.prep['tempDir'] + 'next/', self.config.prep['tempDir'] + 'current/')
        

        def get_preprocess_condition(epoch):
            if epoch == self.config.train['num_epochs']:
                return False
            if not fileExist(self.config.prep['tempDir'] + 'current'):
                return True
            if fileExist(self.config.prep['tempDir'] + 'current/info.json'):
                with open(self.config.prep['tempDir'] + 'current/info.json', 'r') as f:
                    info = json.load(f)
                return \
                    info['dataset_json_path'] != self.config.prep['trainCateDictDir'] or \
                    info['util_percent'] != self.config.prep['util_percent']
            return self.config.prep['prep_every_epoch']
        
        def init_data_loader():
            clippedDataPath = self.config.prep['tempDir'] + 'current/'
            data_dir_list = [[0 if 'fake' in i else 1, clippedDataPath + i] for i in fileWalk(clippedDataPath) if i.endswith('.mp4')]
            self.data_loader = TDM.DataLoader(
                DatasetLoader(data_dir_list, self.config.prep['transform']),
                batch_size=self.config.train['batch_size'],
                shuffle=True,
                num_workers=self.config.train['num_workers'],
                pin_memory=True
            )


        def train(epoch):

            def get_acc(output, sign):
                acc = torch.eq((output>0.5).int(), sign).tolist()
                acc = [i[0] and i[1] for i in acc]
                return sum(acc) / len(acc)


            def save_to_history(out_loss, acc, total_loss=None, gaze_loss=None):
                self.temp_history['out_loss'].append(out_loss)
                self.temp_history['acc'].append(acc)
                if self.use_gaze:
                    self.temp_history['total_loss'].append(total_loss)
                    self.temp_history['gaze_loss'].append(gaze_loss)
            
            
            self.output_lock.acquire()
            print('-' * 50)
            progress_bar = ProgressBar(
                'green',
                'Epoch {}/{}'.format(epoch+1, self.config.train['num_epochs']),
                len(self.data_loader),
            )
            print('-' * 50)
            self.output_lock.release()

            self.temp_history = {
                'out_loss': [],
                'acc': [],
                'total_loss': [],
                'gaze_loss': [],
            }

            self.model.train()
            torch.manual_seed(self.config.train['seed'] + epoch)

            for sign, frames in self.data_loader:
                # Convert variables to correct formats and device
                sign = nn.functional.one_hot(sign.cuda(non_blocking=True), num_classes=2).float()
                sign_var = torch.autograd.Variable(sign).cuda()
                frames_var = torch.autograd.Variable(frames.cuda(non_blocking=True)).cuda()

                output = self.model(frames_var)

                if self.use_gaze:
                    gazeTarget = self.gaze_backend(frames_var.view(-1, 3, 224, 224))
                    gazeEmb = self.model.module.gaze_out.view(-1, gazeTarget.shape[1])

                output = {
                    'out': output,
                    'gaze': gazeEmb if self.use_gaze else None,
                }
                target = {
                    'out': sign_var,
                    'gaze': gazeTarget if self.use_gaze else None
                }

                total_loss = self.loss(output, target)
                acc = get_acc(output['out'][:, :2], sign)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                save_to_history(
                    float(self.loss.out_loss.cpu().detach()),
                    float(acc),
                    float(total_loss.cpu().detach()) if self.use_gaze else None,
                    float(self.loss.gaze_loss.cpu().detach()) if self.use_gaze else None
                )

                progress_bar.Update(
                    '- Mean Out Loss: {:.4f}<br>- Mean Acc: {:.4f}'.format(
                        np.mean(self.temp_history['out_loss']),
                        np.mean(self.temp_history['acc']),
                    ) + ('<br>- Mean Total Loss: {:.4f}<br>- Mean Gaze Loss: {:.4f}'.format(
                        np.mean(self.temp_history['total_loss']),
                        np.mean(self.temp_history['gaze_loss']),
                    ) if self.use_gaze else '')
                )
            
            print(
                '- Mean Out Loss: {:.4f}\n- Mean Acc: {:.4f}'.format(
                    np.mean(self.temp_history['out_loss']),
                    np.mean(self.temp_history['acc']),
                ) + ('\n- Mean Total Loss: {:.4f}\n- Mean Gaze Loss: {:.4f}'.format(
                    np.mean(self.temp_history['total_loss']),
                    np.mean(self.temp_history['gaze_loss']),
                ) if self.use_gaze else '')
            )
            

        def extend_history(temp_history):
            self.history['len_per_epoch'] = len(temp_history['out_loss'])
            self.history['out_loss'].extend(temp_history['out_loss'])
            self.history['acc'].extend(temp_history['acc'])
            if self.use_gaze:
                self.history['total_loss'].extend(temp_history['total_loss'])
                self.history['gaze_loss'].extend(temp_history['gaze_loss'])
        

        def save_checkpoint(epoch):
            checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'history': self.history,
                'config': self.config,
            }
            torch.save(checkpoint, self.config.basic['checkpointDir'] + self.config.basic['tryID'] + '/checkpoint_{}.pth'.format(epoch+1))


        # Create the checkpoint directory
        self.config.basic['tryID'] = datetime.now().strftime("%y%m%d_%H%M%S")
        mkdir(self.config.basic['checkpointDir'] + self.config.basic['tryID'] + '/')
        
        if get_preprocess_condition(0):
            # Preprocess the training dataset
            preprocess_thread = get_preprocess_thread(0)
            preprocess_thread.start()

            # Wait for the first preprocessing to finish
            preprocess_thread.join()
        
            update_dataset()

        init_data_loader()

        # Start training
        for epoch in range(self.config.train['num_epochs']):

            preprocess_condition = get_preprocess_condition(epoch+1)

            if preprocess_condition:
                # Preprocess the training dataset
                preprocess_thread = get_preprocess_thread(epoch+1)
                preprocess_thread.start()

            train(epoch)
            extend_history(self.temp_history)
            PlotHistory(
                self.history,
                self.use_gaze,
                smooth_window_size=math.ceil(512/self.config.train['batch_size']),
                global_size=0.5,
            )
            save_checkpoint(epoch)
            self.scheduler.step()

            if preprocess_condition:
                # Wait for the preprocessing to finish
                preprocess_thread.join()

                update_dataset()
                init_data_loader()
            
        print('\nTraining finished!')