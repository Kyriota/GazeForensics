from Utils.FileOperation import fileWalk
from Utils.Visualization import ProgressBar, PrintBinConfusionMat, PlotResult
from Core.DatasetLoader import DatasetLoader
from Core.Loss import TestLoss
from Models.MainModel import InitModel
from Config import Config

import torch
from torch import nn
import torch.utils.data as TDM # torch data module

import numpy as np
import json
from collections import Counter
import time



class EvaluateManager:

    def __init__(self, config:Config):

        def get_shrinked_dataset():
            data_dir_count = [i for i in fileWalk(config.test['testClipsDir']) if i.endswith('.mp4')]
            to_delete_num = int(len(data_dir_count) * (1 - config.test['util_percent']))
            for i in range(len(data_dir_count)):
                data_dir_count[i] = data_dir_count[i][:-len(data_dir_count[i].split('_')[-1])-1]
            # Use Counter to count the number of each category
            data_dir_count = [[i[0], i[1], i[1]] for i in Counter(data_dir_count).most_common()]
            # Shrink videos that have most clips
            for _ in range(to_delete_num):
                data_dir_count[0][1] -= 1
                if data_dir_count[0][1] == 0:
                    data_dir_count.pop(0)
                data_dir_count.sort(key=lambda x: x[1], reverse=True)
            data_dir_list = []
            np.random.seed(int(config.test['util_percent'] * 1e6))
            for i in data_dir_count:
                if i[1] == i[2]:
                    temp_dir_list = [[0 if 'fake' in i[0] else 1, config.test['testClipsDir'] + i[0] + '_' + str(j) + '.mp4'] for j in range(i[2])]
                else:
                    temp_dir_list = [[0 if 'fake' in i[0] else 1, config.test['testClipsDir'] + i[0] + '_' + str(j) + '.mp4'] for j in np.random.choice(i[2], i[1], replace=False)]
                data_dir_list.extend(temp_dir_list)
            return data_dir_list


        self.config = config

        # Init dataset list
        if config.test['util_percent'] < 1:
            data_dir_list = get_shrinked_dataset()
        else:
            data_dir_list = [[0 if 'fake' in i else 1, config.test['testClipsDir'] + i] for i in fileWalk(config.test['testClipsDir']) if i.endswith('.mp4')]
            
        # Init DataLoader
        self.data_loader = TDM.DataLoader(
            DatasetLoader(data_dir_list, config.prep['transform']),
            batch_size=config.test['batch_size'],
            shuffle=True,
            num_workers=config.test['num_workers'],
            pin_memory=True
        )


    
    def evaluate(self, checkpoint_path, show_progress=True, show_confusion_mat=True):

        def single_shot(pth_path, epoch=1, epochs=1):
            checkpoint = torch.load(pth_path)

            # Init model
            model = InitModel(checkpoint['config'])
            model.load_state_dict(checkpoint['state_dict'])
            model.cuda()
            model.eval()
            
            # Init loss function
            loss_func = TestLoss()

            # Init result list
            preds = []
            ground_truths = []
            losses = []
            corrects = []

            if show_progress:
                # Init progress bar
                print('-' * 50)
                progress_bar = ProgressBar(
                    'yellow',
                    'Epoch {}/{}'.format(epoch, epochs),
                    len(self.data_loader),
                )
                print('-' * 50)
            
            # Evaluate
            with torch.no_grad():
                for sign, frames in self.data_loader:
                    _sign = nn.functional.one_hot(sign, num_classes=2).float().cuda(non_blocking=True)
                    _frames = frames.cuda(non_blocking=True)

                    output = model(_frames)
                    pred = output[:, :2]

                    loss = loss_func(pred, _sign)
                    pred = (pred>0.5)[:, 1:].squeeze().tolist()
                    correct = [i==j for i, j in zip(pred, sign.tolist())]

                    preds.extend(pred)
                    corrects.extend(correct)
                    losses.append(float(loss.cpu().detach()))
                    ground_truths.extend([bool(i) for i in sign.tolist()])

                    if show_progress:
                        progress_bar.Update('- Mean Out Loss: {:.4f}<br>- Mean Acc: {:.4f}'.format(
                            np.mean(losses),
                            sum(corrects) / len(corrects)
                        ))
            
            print('Time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            if show_confusion_mat:
                preds = np.array(preds)
                ground_truths = np.array(ground_truths)
                confusion_mat = PrintBinConfusionMat(preds, ground_truths)

            acc = sum(corrects) / len(corrects)
            loss = np.mean(losses)

            print(' > Acc: {:.4f} | Loss: {:.4f}'.format(acc, loss))

            return acc, loss, confusion_mat
        
        
        def get_train_result(checkpoint_path):
            train_history = torch.load(checkpoint_path)['history']
            len_per_epoch = train_history['len_per_epoch']
            train_result = {
                'out_loss': [
                    np.mean(
                        train_history['out_loss'][i:i+len_per_epoch]
                    ) for i in range(0, len(train_history['out_loss']), len_per_epoch)
                ],
                'acc': [
                    np.mean(
                        train_history['acc'][i:i+len_per_epoch]
                    ) for i in range(0, len(train_history['acc']), len_per_epoch)
                ]
            }
            return train_result
        

        if checkpoint_path.endswith('.pth'):
            acc, loss, _ = single_shot(checkpoint_path)
            return acc, loss
        
        checkpoint_paths = fileWalk(checkpoint_path)
        # Sort checkpoint paths
        checkpoint_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        accs = []
        losses = []
        confusion_mats = []
        for epoch, pth_path in enumerate(checkpoint_paths):
            acc, loss, confusion_mat = single_shot(checkpoint_path + pth_path, epoch+1, len(checkpoint_paths))
            accs.append(acc)
            losses.append(loss)
            confusion_mats.append(confusion_mat)
        
        result = {
            'test_result': {'acc': accs, 'out_loss': losses, 'confusion_mat': confusion_mats},
            'train_result': get_train_result(checkpoint_path + checkpoint_paths[-1]),
        }
        with open(self.config.test['resultDir'] + self.config.basic['tryID'] + '.json', 'w') as f:
            json.dump(result, f)
        PlotResult(
            result,
            result_path=self.config.test['resultDir'] + self.config.basic['tryID'] + '.png'
        )

        return accs, losses