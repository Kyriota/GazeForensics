from Models.MainModel import InitModel, ImageDecoder
from Models.GazeBackend import GazeBackend
from Utils.FileOperation import mkdir, fileWalk, ls, fileExist
from Utils.Visualization import ProgressBar, PlotHistory, smooth_data, sample_data, show_decoder_samples
from Preprocess.DatasetClipping import ClipTrainDatasetParallel
from Core.DatasetLoader import DatasetLoader
from Core.Loss import TrainLoss
from Config import Config
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.utils.data as TDM # torch data module
from torchvision.models.resnet import BasicBlock

import threading
from datetime import datetime
import time
import numpy as np



def init_optimizer(para, lr, weight_decay):
    return torch.optim.Adam(para, lr=lr, weight_decay=weight_decay)



def init_scheduler(optimizer, step_size, gamma):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
        last_epoch=-1,
        verbose=True
    )



class TrainManager:

    def __init__(self, config:Config):

        self.config = config
        self.gaze_backend = None
        self.data_loader = None
        self.current_DS_path = None
        self.output_lock = threading.Lock()
        self.use_gaze = self.config.loss['gaze_weight'] > 0.0
        self.use_decoder = self.config.loss['decoder_weight'] > 0.0
        self.enable_decoder = False
        self.temp_history = None
        self.num_epochs = self.config.train['num_epochs']
        self.history = {
            'out_loss': [],
            'acc': [],
            'emb_loss': [],
            'total_loss': [],
            'len_per_epoch': None,
        }

        # Init the model
        self.model = InitModel(config)
        if self.use_decoder:
            self.decoder = ImageDecoder().cuda()

        # Init optimizer, scheduler and loss function
        self.optimizer = init_optimizer(
            self.model.parameters(),
            lr=config.opti['lr'],
            weight_decay=config.opti['weight_decay']
        )
        self.scheduler = init_scheduler(
            self.optimizer,
            step_size=config.opti['step_size'],
            gamma=config.opti['gamma']
        )
        if config.loss['loss_func'] == 'standard':
            self.loss = TrainLoss(config.loss)
        else:
            raise ValueError('Invalid loss function: ' + config.loss['loss_func'])

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
            if epoch == self.num_epochs:
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

            def get_rec_img(frames):
                with torch.no_grad():
                    _ = self.model(frames)
                    emb_out = self.model.module.emb_out.view(-1, self.config.model['emb_dim'])
                return self.decoder(emb_out)


            def train_decoder():
                # Reset Temp History
                extend_history(self.temp_history)
                init_temp_history()

                temp_decoder_loss_history = []
                
                ######################
                #    Train Decoder   #
                ######################

                for _ in range(self.config.train['decoder_num_epochs']):
                    for sign, frames in self.data_loader:

                        frames = frames.cuda(non_blocking=True)

                        self.decoder.zero_grad()

                        rec_img = get_rec_img(frames)
                        
                        decoder_loss = self.loss.MSE(rec_img, frames.view(-1, 3, 224, 224))

                        self.decoder_optimizer.zero_grad()
                        decoder_loss.backward()
                        self.decoder_optimizer.step()

                        temp_decoder_loss_history.append(float(decoder_loss.cpu().detach()))

                        decoder_progress_bar.Update(
                            '- {}/{}<br>'.format(
                                decoder_epoch,
                                int(1 / self.config.train['decoder_alpha']),
                            ) + '- Mean Decoder Loss: {:.4f}'.format(
                                np.mean(temp_decoder_loss_history),
                            )
                        )

                print(
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    '\n- Mean Decoder Loss: {:.4f}'.format(
                        np.mean(temp_decoder_loss_history),
                    )
                )


            def get_acc(output, sign):
                acc = torch.eq((output>0.5).int(), sign).tolist()
                acc = [i[0] and i[1] for i in acc]
                return sum(acc) / len(acc)


            def save_to_history(
                    out_loss,
                    acc,
                    total_loss=None,
                    emb_loss=None,
                    decoder_loss=None,
                ):
                self.temp_history['out_loss'].append(out_loss)
                self.temp_history['acc'].append(acc)
                if self.use_gaze:
                    self.temp_history['total_loss'].append(total_loss)
                    self.temp_history['emb_loss'].append(emb_loss)
                if self.use_decoder:
                    self.temp_history['decoder_loss'].append(decoder_loss)
            
            
            self.output_lock.acquire()
            print('-' * 50)
            progress_bar = ProgressBar(
                'green',
                'Train {}/{}'.format(epoch+1, self.num_epochs),
                len(self.data_loader),
            )
            if self.enable_decoder:
                decoder_progress_bar = ProgressBar(
                    'green',
                    'Decoder',
                    len(self.data_loader) * self.config.train['decoder_num_epochs'],
                )
            print('-' * 50)
            self.output_lock.release()

            init_temp_history()

            self.model.train()
            self.decoder.train()
            torch.manual_seed(self.config.train['seed'] + epoch)

            if self.enable_decoder:
                decoder_epoch = 0

            for train_i, (sign, frames) in enumerate(self.data_loader):
            
                if self.enable_decoder and train_i % (int(self.config.train['decoder_alpha'] * len(self.data_loader))) == 0:

                    decoder_epoch += 1
                    self.model.eval()
                    decoder_progress_bar.progress = 0
                    
                    # Show Decoder Output Samples
                    show_decoder_samples(
                        sample_data(frames),
                        sample_data(get_rec_img(frames).view(-1, 14, 3, 224, 224)),
                        title='Before Decoder Training',
                    )

                    train_decoder()

                    # Show Decoder Output Samples
                    show_decoder_samples(
                        sample_data(frames),
                        sample_data(get_rec_img(frames).view(-1, 14, 3, 224, 224)),
                        title='After Decoder Training',
                    )
                    print('-' * 50)

                    self.model.train()

                # Convert variables to correct formats and device
                sign = nn.functional.one_hot(sign.cuda(non_blocking=True), num_classes=2).float()
                frames = frames.cuda(non_blocking=True)
                
                ######################
                #  Train Main Model  #
                ######################
                self.model.zero_grad()
                self.decoder.zero_grad()

                output = self.model(frames)

                if self.use_gaze:
                    emb_tar = self.gaze_backend(frames.view(-1, 3, 224, 224))
                    emb_out = self.model.module.emb_out.view(-1, emb_tar.shape[1])

                if self.enable_decoder:
                    # Reconstruct Image
                    rec_img = self.decoder(self.model.module.emb_out.view(-1, self.config.model['emb_dim']))

                output = {
                    'out': output,
                    'emb_out': emb_out if self.use_gaze else None,
                    'decoder': rec_img if self.enable_decoder else None,
                }
                target = {
                    'out': sign,
                    'emb_out': emb_tar if self.use_gaze else None,
                    'decoder': frames.view(-1, 3, 224, 224) if self.enable_decoder else None,
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
                    float(self.loss.emb_loss.cpu().detach()) if self.use_gaze else None,
                    float(self.loss.decoder_loss.cpu().detach()) if self.enable_decoder else None,
                )

                progress_bar.Update(
                    '- Mean Out Loss: {:.4f}<br>- Mean Acc: {:.4f}'.format(
                        np.mean(self.temp_history['out_loss']),
                        np.mean(self.temp_history['acc']),
                    ) + (
                        '<br>- Mean Total Loss: {:.4f}<br>- Mean Emb Loss: {:.4f}'.format(
                        np.mean(self.temp_history['total_loss']),
                        np.mean(self.temp_history['emb_loss']),
                    ) if self.use_gaze else '') + (
                        '<br>- Mean Decoder Loss: {:.4f}'.format(
                        np.mean(self.temp_history['decoder_loss']),
                    ) if self.enable_decoder else '')
                )
            
            print(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                '\n- Mean Out Loss: {:.4f}\n- Mean Acc: {:.4f}'.format(
                    np.mean(self.temp_history['out_loss']),
                    np.mean(self.temp_history['acc']),
                ) + ('\n- Mean Total Loss: {:.4f}\n- Mean Emb Loss: {:.4f}'.format(
                    np.mean(self.temp_history['total_loss']),
                    np.mean(self.temp_history['emb_loss']),
                ) if self.use_gaze else '') + (
                    '\n- Mean Decoder Loss: {:.4f}'.format(
                    np.mean(self.temp_history['decoder_loss']),
                ) if self.enable_decoder else '')
            )

            extend_history(self.temp_history)

        
        def init_temp_history():
            self.temp_history = {
                'out_loss': [],
                'acc': [],
                'total_loss': [],
                'emb_loss': [],
                'decoder_loss': [],
            }
            

        def extend_history(temp_history):
            self.history['len_per_epoch'] = len(temp_history['out_loss'])
            self.history['out_loss'].extend(temp_history['out_loss'])
            self.history['acc'].extend(temp_history['acc'])
            if self.use_gaze:
                self.history['total_loss'].extend(temp_history['total_loss'])
                self.history['emb_loss'].extend(temp_history['emb_loss'])
        

        def save_checkpoint(epoch):
            checkpoint = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'history': self.history,
                'config': self.config,
                'decoder_dict': self.decoder.state_dict() if self.use_decoder else None,
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
        for epoch in range(self.num_epochs):
            
            dataset_epoch = epoch+1 if self.config.prep['prep_every_epoch'] else 0
            next_folder_name = get_dataset_folder_name(dataset_epoch)
            preprocess_condition = get_preprocess_condition(dataset_epoch, next_folder_name)

            if preprocess_condition:
                # Preprocess the training dataset
                preprocess_thread = get_preprocess_thread(dataset_epoch, self.config.prep['tempDir'] + next_folder_name + '/')
                preprocess_thread.start()

            train(epoch)
            save_checkpoint(epoch)
            PlotHistory(
                self.history,
                self.use_gaze,
                slice_num=100,
                global_size=0.5,
            )

            self.scheduler.step()
            if self.enable_decoder:
                self.decoder_scheduler.step()
            
            if self.use_decoder and epoch==self.config.train['decoder_enable_epoch']:
                self.decoder_optimizer = init_optimizer(
                    self.decoder.parameters(),
                    lr=self.config.opti['decoder_lr'],
                    weight_decay=self.config.opti['weight_decay']
                )
                self.decoder_scheduler = init_scheduler(
                    self.decoder_optimizer,
                    step_size=self.config.opti['step_size'],
                    gamma=self.config.opti['gamma']
                )
                self.enable_decoder = True
                self.model.module.save_emb_out = True
                self.loss.enable_decoder = True

            if preprocess_condition:
                # Wait for the preprocessing to finish
                preprocess_thread.join()

            if epoch != self.num_epochs - 1:
                # Reinitialize the data loader
                self.current_DS_path = self.config.prep['tempDir'] + get_dataset_folder_name(epoch+1) + '/'
                init_data_loader()
            
        print('\nTraining finished!')
        return 0