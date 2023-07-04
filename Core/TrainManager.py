from Models.MainModel import InitModel
from Models.GazeBackend import GazeBackend
from Utils.FileOperation import mkdir, fileWalk, ls, fileExist
from Utils.Visualization import ProgressBar, PlotHistory
from Preprocess.DatasetClipping import ClipTrainDatasetParallel
from Core.DatasetLoader import DatasetLoader
from Core.Loss import TrainLoss, CustomLoss
from Config import Config

import torch
from torch import nn
import torch.utils.data as TDM # torch data module
from torchvision.models.resnet import BasicBlock

import threading
from datetime import datetime
import time
import numpy as np



class TrainManager:

    def __init__(self, config:Config):

        self.config = config
        self.gaze_backend = None
        self.data_loader = None
        self.current_DS_path = None
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
            self.gaze_backend = GazeBackend(
                BasicBlock,
                [2, 2, 2, 2],
                emb_dim=self.config.model['emb_dim'],
            )
            backend_name = 'gazeBackend_' + str(self.config.model['emb_dim']) + '.pkl'
            state_dict = torch.load(self.config.basic['checkpointDir'] + backend_name)
            print(self.gaze_backend.load_state_dict(state_dict, strict=False))
            self.gaze_backend = nn.DataParallel(self.gaze_backend).cuda()
            self.gaze_backend.eval()
    
    
    def train(self):

        def get_preprocess_thread(epoch, DS_save_path):
            return threading.Thread(
                target=ClipTrainDatasetParallel,
                kwargs={
                    'json_path': self.config.prep['trainCateDictDir'],
                    'output_path': DS_save_path,
                    'output_lock': self.output_lock,
                    'thread_num': self.config.prep['thread_num'],
                    'util_percent': self.config.prep['util_percent'],
                    'epoch': epoch,
                }
            )
        

        def get_preprocess_condition(epoch, target_folder_name):
            # Check if the dataset is already preprocessed
            #  - Returning True means preprocess is needed
            #  - Returning False means preprocess is not needed
            if epoch == self.config.train['num_epochs']:
                return False
            saved_datasets = ls(self.config.prep['tempDir'])
            for folder_name in saved_datasets:
                if folder_name == target_folder_name:
                    return False
            return True
        

        def get_dataset_folder_name(epoch):
            return \
                self.config.basic['train_DS_name'] + '_' + \
                '{:.1%}'.format(self.config.prep['util_percent']) + '_' + \
                'epoch' + str(epoch)

        
        def init_data_loader():
            clippedDataPath = self.current_DS_path
            print('[*] Loading dataset from: ' + clippedDataPath)
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
                    with torch.no_grad():
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
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                '\n- Mean Out Loss: {:.4f}\n- Mean Acc: {:.4f}'.format(
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
        if self.config.basic['tryID'] is None:
            self.config.basic['tryID'] = datetime.now().strftime("%y%m%d_%H%M%S")
        if fileExist(self.config.basic['checkpointDir'] + self.config.basic['tryID'] + '/'):
            print('>> ERROR: Checkpoint directory already exists! Please change tryID in config.')
            return -1
        mkdir(self.config.basic['checkpointDir'] + self.config.basic['tryID'] + '/')
        
        current_folder_name = get_dataset_folder_name(0)
        self.current_DS_path = self.config.prep['tempDir'] + current_folder_name + '/'
        if get_preprocess_condition(0, current_folder_name):
            # Preprocess the training dataset
            preprocess_thread = get_preprocess_thread(0, self.current_DS_path)
            preprocess_thread.start()

            # Wait for the first preprocessing to finish
            preprocess_thread.join()

        init_data_loader()

        # Start training
        for epoch in range(self.config.train['num_epochs']):
            
            dataset_epoch = epoch+1 if self.config.prep['prep_every_epoch'] else 0
            next_folder_name = get_dataset_folder_name(dataset_epoch)
            preprocess_condition = get_preprocess_condition(dataset_epoch, next_folder_name)

            if preprocess_condition:
                # Preprocess the training dataset
                preprocess_thread = get_preprocess_thread(dataset_epoch, self.config.prep['tempDir'] + next_folder_name + '/')
                preprocess_thread.start()

            train(epoch)
            extend_history(self.temp_history)
            save_checkpoint(epoch)
            PlotHistory(
                self.history,
                self.use_gaze,
                slice_num=100,
                global_size=0.5,
            )
            self.scheduler.step()

            if preprocess_condition:
                # Wait for the preprocessing to finish
                preprocess_thread.join()

            if epoch != self.config.train['num_epochs'] - 1:
                # Reinitialize the data loader
                self.current_DS_path = self.config.prep['tempDir'] + get_dataset_folder_name(epoch+1) + '/'
                init_data_loader()
            
        print('\nTraining finished!')
        return 0