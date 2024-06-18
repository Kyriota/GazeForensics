from Models.MainModel import InitModel
from Models.GazeBackend import GazeBackend
from Utils.FileOperation import mkdir, fileWalk, ls, fileExist
from Utils.Visualization import ProgressBar, PlotHistory, NormalizeImage
from Preprocess.DatasetClipping import ClipTrainDatasetParallel
from Core.DatasetLoader import DatasetLoader
from Core.Loss import TrainLoss, smooth_label
from Config import Config
from ConfigTypes import TransformType

import torch
from torch import nn
import torch.utils.data as TDM  # torch data module
from torchvision.models.resnet import BasicBlock

import threading
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class TrainManager:

    def __init__(self, config: Config):

        self.config = config
        self.gaze_backend = None
        self.data_loader = None
        self.current_DS_path = None
        self.output_lock = threading.Lock()
        self.use_gaze = self.config.loss.gaze_weight > 0.0
        self.use_argument = self.config.prep.argument
        self.rand_horizontal_flip = self.config.prep.rand_horizontal_flip
        self.temp_history = None
        self.lastEpoch = 0
        self.history = {
            'out_loss': [],
            'acc': [],
            'gaze_loss': [],
            'total_loss': [],
            'len_per_epoch': None,
        }
        self.grad_norm_history = []

        # Init the model
        self.model = InitModel(config)

        # Init optimizer and loss function
        self.need_backend_opti = not self.model.module.freeze_backend or self.model.module.leaky_dim > 0
        if self.need_backend_opti:
            self.backend_optimizer = torch.optim.AdamW(
                [
                    {'params': self.model.module.base_model.parameters()},
                    {'params': self.model.module.gaze_fc.parameters()},
                ],
                lr=config.opti.lr,
                weight_decay=config.opti.weight_decay
            )
        self.classifier_optimizer = torch.optim.AdamW(
            [
                {'params': self.model.module.MHA_Q_fc.parameters()},
                {'params': self.model.module.MHA_K_fc.parameters()},
                {'params': self.model.module.MHA_V_fc.parameters()},
                {'params': self.model.module.multihead_attn.parameters()},
                {'params': self.model.module.MHA_comp.parameters()},
                {'params': self.model.module.last_fc.parameters()},
            ],
            lr=config.opti.lr,
            weight_decay=config.opti.weight_decay
        )
        if self.need_backend_opti:
            self.backend_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.backend_optimizer,
                max_lr=config.opti.lr,
                pct_start=config.opti.backend_pct_start,
                div_factor=config.opti.backend_div_factor,
                total_steps=config.train.num_epochs,
            )
        self.classifier_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.classifier_optimizer,
            max_lr=config.opti.lr,
            pct_start=config.opti.classifier_pct_start,
            div_factor=config.opti.classifier_div_factor,
            total_steps=config.train.num_epochs,
        )
        self.loss = TrainLoss(config.loss)

        # Load checkpoint
        if self.config.model.checkpoint:
            checkpoint = torch.load(self.config.model.checkpoint)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.lastEpoch = checkpoint['epoch'] + 1
            self.history = checkpoint['history']

        if self.use_gaze:
            # # Init gaze backend
            self.gaze_backend = GazeBackend(
                BasicBlock,
                [2, 2, 2, 2],
                emb_dim=self.config.model.emb_dim,
            )
            state_dict = torch.load(self.config.model.gaze_backend_path)
            self.train_print(self.gaze_backend.load_state_dict(state_dict, strict=False))
            self.gaze_backend = nn.DataParallel(self.gaze_backend).cuda()
            self.gaze_backend.eval()

    def train_print(self, content, end='\n'):
        if self.config.train.verbose:
            print(content, end=end)

    def init_data_loader(self):
        self.train_print('[*] Loading dataset from: ' + self.current_DS_path)
        data_dir_list = [[0 if 'fake' in i else 1, self.current_DS_path + i]
                         for i in fileWalk(self.current_DS_path) if i.endswith('.mp4')]
        self.data_loader = TDM.DataLoader(
            DatasetLoader(
                data_dir_list,
                self.config.prep.transform,
                use_argument=self.use_argument,
                rand_horizontal_flip=self.rand_horizontal_flip,
            ),
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.num_workers,
            pin_memory=False
        )

    def train(self):

        def get_preprocess_thread(epoch, DS_save_path):
            return threading.Thread(
                target=ClipTrainDatasetParallel,
                kwargs={
                    'json_path': self.config.prep.trainCateDictDir,
                    'output_path': DS_save_path,
                    'output_lock': self.output_lock,
                    'thread_num': self.config.prep.thread_num,
                    'util_percent': self.config.prep.util_percent,
                    'epoch': epoch,
                }
            )

        def get_preprocess_condition(epoch, target_folder_name):
            # Check if the dataset is already preprocessed
            #  - Returning True means preprocess is needed
            #  - Returning False means preprocess is not needed
            if epoch == self.config.train.num_epochs:
                return False
            saved_datasets = ls(self.config.prep.tempDir)
            for folder_name in saved_datasets:
                if folder_name == target_folder_name:
                    return False
            return True

        def get_dataset_folder_name(epoch):
            return \
                self.config.basic.train_DS_name + '_' + \
                '{:.1%}'.format(self.config.prep.util_percent) + '_' + \
                'epoch' + str(epoch)

        def train(epoch):

            def get_acc(output, sign):
                acc = torch.eq((output > 0.5).int(), sign).tolist()
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
                'Epoch {}/{}'.format(epoch+1, self.config.train.num_epochs),
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
            torch.manual_seed(self.config.train.seed + epoch)

            for sign, frames in self.data_loader:
                # Convert variables to correct formats and device
                sign = nn.functional.one_hot(sign.cuda(non_blocking=True), num_classes=2).float()
                sign_var = torch.autograd.Variable(smooth_label(
                    sign,
                    self.config.train.smooth_label_alpha,
                    self.config.train.smooth_label_beta,
                )).cuda()
                if self.use_argument:
                    frames_var = torch.autograd.Variable(frames[TransformType.argument].cuda(non_blocking=True)).cuda()
                else:
                    frames_var = torch.autograd.Variable(frames[TransformType.standard].cuda(non_blocking=True)).cuda()

                # # Debug code: Show frames_var using PIL.Image
                # for debug_i in range(frames_var.shape[0]):
                #     for debug_j in range(frames_var[debug_i].shape[0]):
                #         img = frames_var[debug_i][debug_j].cpu().detach().numpy()
                #         img = np.transpose(img, (1, 2, 0))
                #         img = Image.fromarray(NormalizeImage(img))
                #         img.show()
                # input()

                output = self.model(frames_var)

                if self.use_gaze:
                    with torch.no_grad():
                        frames_var_gaze = torch.autograd.Variable(frames[TransformType.standard].cuda(non_blocking=True)).cuda()
                        gazeTarget = self.gaze_backend(frames_var_gaze.view(-1, 3, 224, 224))
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

                if self.need_backend_opti:
                    self.backend_optimizer.zero_grad()
                self.classifier_optimizer.zero_grad()
                total_loss.backward()

                if self.config.opti.enable_grad_clip:
                    # Store current gradient norm
                    grad_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.detach().data.norm(2).item() ** 2
                    self.grad_norm_history.append(grad_norm ** 0.5)

                    # Get median of recent gradient norm
                    grad_norm_median = np.median(self.grad_norm_history[-self.config.opti.grad_clip_ref_range:])
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_norm_median * self.config.opti.grad_clip_mul)

                if self.need_backend_opti:
                    self.backend_optimizer.step()
                self.classifier_optimizer.step()

                save_to_history(
                    float(self.loss.out_loss.cpu().detach()),
                    float(acc),
                    float(total_loss.cpu().detach()
                          ) if self.use_gaze else None,
                    float(self.loss.gaze_loss.cpu().detach()
                          ) if self.use_gaze else None
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

            self.train_print(
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
                'history': self.history,
                'config': self.config,
            }
            torch.save(checkpoint, self.config.basic.checkpointDir +
                       self.config.basic.tryID + '/checkpoint_{}.pth'.format(epoch+1))

        # Create the checkpoint directory
        if fileExist(self.config.basic.checkpointDir + self.config.basic.tryID + '/'):
            print(
                '>> ERROR: Checkpoint directory already exists! Please change tryID in config.')
            return -1
        mkdir(self.config.basic.checkpointDir + self.config.basic.tryID + '/')

        current_folder_name = get_dataset_folder_name(self.lastEpoch + 0)
        self.current_DS_path = self.config.prep.tempDir + current_folder_name + '/'
        if get_preprocess_condition(0, current_folder_name):
            # Preprocess the training dataset
            preprocess_thread = get_preprocess_thread(self.lastEpoch + 0, self.current_DS_path)
            preprocess_thread.start()

            # Wait for the first preprocessing to finish
            preprocess_thread.join()

        self.init_data_loader()

        # Start training
        for epoch in range(self.config.train.num_epochs-self.lastEpoch):

            dataset_epoch = self.lastEpoch+epoch + 1 if self.config.prep.prep_every_epoch else self.lastEpoch+0
            next_folder_name = get_dataset_folder_name(dataset_epoch)
            preprocess_condition = get_preprocess_condition(dataset_epoch, next_folder_name)

            if preprocess_condition:
                # Preprocess the training dataset
                preprocess_thread = get_preprocess_thread(dataset_epoch, self.config.prep.tempDir + next_folder_name + '/')
                preprocess_thread.start()

            train(self.lastEpoch+epoch)
            extend_history(self.temp_history)
            save_checkpoint(self.lastEpoch+epoch)
            if self.config.train.verbose:
                PlotHistory(
                    self.history,
                    self.use_gaze,
                    slice_num=100,
                    global_size=0.5,
                )

            if self.need_backend_opti:
                self.backend_scheduler.step()
            self.classifier_scheduler.step()

            if self.use_gaze:
                self.loss.gaze_weight_schedule()

            if preprocess_condition:
                # Wait for the preprocessing to finish
                preprocess_thread.join()

            if self.lastEpoch+epoch != self.config.train.num_epochs - 1:
                # Reinitialize the data loader
                self.current_DS_path = self.config.prep.tempDir + get_dataset_folder_name(self.lastEpoch+epoch+1) + '/'
                self.init_data_loader()

        self.train_print('\nTraining finished!')
        return 0
