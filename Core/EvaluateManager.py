from Utils.FileOperation import fileWalk
from Utils.Visualization import ProgressBar, GetBinConfusionMat, PlotResult
from Core.DatasetLoader import DatasetLoader
from Core.Loss import TestLoss, smooth_label
from Models.MainModel import InitModel
from Config import Config

import torch
from torch import nn
import torch.utils.data as TDM # torch data module

import numpy as np
import json
from collections import Counter
import time
import re



class EvaluateManager:

    def __init__(self, config:Config):

        def get_shrinked_dataset():
            data_dir_count = [i for i in fileWalk(config.test['testClipsDir']) if i.endswith('.mp4')]
            to_delete_num = int(len(data_dir_count) * (1 - config.test['util_percent']))
            for i in range(len(data_dir_count)):
                data_dir_count[i] = data_dir_count[i][:-len(data_dir_count[i].split('_')[-1])-1]
            # Use Counter to count the number of each category
            data_dir_count = [[i[0], i[1], i[1]] for i in Counter(data_dir_count).most_common()]
            #   data_dir_count: [[category, shrinked_num, total_num], ...]
            #   e.g. ['real_test/real_test_13_1', 1498, 1498]
            # Shrink videos that have most clips
            for _ in range(to_delete_num):
                data_dir_count[0][1] -= 1
                if data_dir_count[0][1] == 0:
                    raise Exception('There is no enough data to test, please increase the utilization percentage')
                data_dir_count.sort(key=lambda x: x[1], reverse=True)
            data_dir_list = []
            np.random.seed(int(config.test['util_percent'] * 1e6))
            for i in data_dir_count:
                if i[1] == i[2]:
                    # If the number of clips is equal to the number of clips after shrink
                    #  then use all clips
                    temp_dir_list = [[0 if 'fake' in i[0] else 1, config.test['testClipsDir'] + i[0] + '_' + str(j) + '.mp4'] for j in range(i[2])]
                else:
                    # Else, randomly choose clips
                    temp_dir_list = [[0 if 'fake' in i[0] else 1, config.test['testClipsDir'] + i[0] + '_' + str(j) + '.mp4'] for j in np.random.choice(i[2], i[1], replace=False)]
                data_dir_list.extend(temp_dir_list)
            return data_dir_list
        

        def natural_keys(text):
            '''
            alist.sort(key=natural_keys) sorts in human order
            http://nedbatchelder.com/blog/200712/human_sorting.html
            (See Toothy's implementation in the comments)
            '''
            def atoi(text):
                return int(text) if text.isdigit() else text
            
            return [atoi(c) for c in re.split(r'(\d+)', text)]


        self.config = config

        # Init dataset list
        if config.test['util_percent'] < 1:
            self.data_dir_list = get_shrinked_dataset()
        else:
            self.data_dir_list = [[0 if 'fake' in i else 1, config.test['testClipsDir'] + i] for i in fileWalk(config.test['testClipsDir']) if i.endswith('.mp4')]
        self.data_dir_list.sort(key=lambda x: natural_keys(x[1]))
            
        # Init DataLoader
        self.data_loader = TDM.DataLoader(
            DatasetLoader(self.data_dir_list, self.config.test['transform'], return_path=True),
            batch_size=self.config.test['batch_size'],
            shuffle=False,
            num_workers=self.config.test['num_workers'],
            pin_memory=True
        )
    

    def eval_print(self, content, end='\n'):
        if self.config.test['verbose']:
            print(content, end=end)


    
    def evaluate(
            self,
            checkpoint_path,
            show_progress=True,
            show_confusion_mat=True,
        ):

        def single_shot(pth_path, epoch=1, epochs=1):

            def get_parent_id(path):
                path = path[len(self.config.test['testClipsDir']):-4]
                last_element_len = len(path.split('_')[-1])
                return path[:-last_element_len-1]
            
            
            checkpoint = torch.load(pth_path)

            # Version control
            if 'freeze_backend' not in checkpoint['config'].model.keys():
                checkpoint['config'].model['freeze_backend'] = False

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
            parent_ids = []
            ids = []
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
                for sign, frames, vid_paths in self.data_loader:
                    _sign = nn.functional.one_hot(sign, num_classes=2).float().cuda(non_blocking=True)
                    _sign = smooth_label(
                        _sign,
                        self.config.train['smooth_label_alpha']+0.5*self.config.train['smooth_label_beta'],
                        0,
                    )
                    _frames = frames['standard'].cuda(non_blocking=True)

                    output = model(_frames)
                    pred = output[:, :2]
                    # pred = torch.softmax(output[:, :2], dim=1)

                    loss = loss_func(pred, _sign)
                    pred = pred[:, 1:].squeeze(dim=1).tolist()
                    correct = [(i[0] > 0.5) == (i[1] == 1) for i in zip(pred, sign.tolist())]

                    ground_truths.extend([bool(i) for i in sign.tolist()])
                    preds.extend(pred)
                    corrects.extend(correct)
                    losses.append(float(loss.cpu().detach()))
                    parent_ids.extend([get_parent_id(path) for path in vid_paths])
                    ids.extend([path[len(self.config.test['testClipsDir']):-4] for path in vid_paths])

                    if show_progress:
                        progress_bar.Update('- Mean Out Loss: {:.4f}<br>- Mean Acc: {:.4f}'.format(
                            np.mean(losses),
                            sum(corrects) / len(corrects),
                        ))
            
            self.eval_print('Time:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

            # Calculate video level accuracy
            vid_preds = []
            vid_corrects = []
            vid_ground_truths = []
            current_vid_preds = []
            vid_ids = []
            for i in range(len(preds)):
                current_vid_preds.append(preds[i])
                if i == len(preds) - 1 or parent_ids[i] != parent_ids[i+1]:
                    vid_pred = np.mean(current_vid_preds)
                    vid_preds.append(vid_pred)
                    vid_corrects.append((vid_pred > 0.5) == ground_truths[i])
                    vid_ground_truths.append(ground_truths[i])
                    vid_ids.append(parent_ids[i])
                    current_vid_preds = []
            
            vid_acc = sum(vid_corrects) / len(vid_corrects)
            acc = sum(corrects) / len(corrects)
            loss = np.mean(losses)

            if show_confusion_mat:
                confusion_mat = [
                    GetBinConfusionMat(
                        np.array(vid_preds) > 0.5,
                        np.array(vid_ground_truths),
                        self.config.test['verbose'],
                    ),
                    GetBinConfusionMat(
                        np.array(preds) > 0.5,
                        np.array(ground_truths),
                        self.config.test['verbose'],
                    ),
                ]

            self.eval_print(' > Vid Acc: {:.4f} | Seq Acc: {:.4f} | Loss: {:.4f}'.format(vid_acc, acc, loss))
            
            return vid_acc, acc, loss, confusion_mat, list(zip(vid_ids, vid_corrects, vid_preds)), list(zip(ids, corrects, preds))
        
        

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
            vid_acc, acc, loss, _, detail_vid, detail_seq = single_shot(checkpoint_path)
            return vid_acc, acc, loss, detail_vid, detail_seq
        
        checkpoint_paths = fileWalk(checkpoint_path)
        # Sort checkpoint paths
        checkpoint_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        if self.config.test['onlyEvalLastN'] is not None:
            checkpoint_paths = checkpoint_paths[-self.config.test['onlyEvalLastN']:]
        vid_accs = []
        accs = []
        losses = []
        confusion_mats = []
        for epoch, pth_path in enumerate(checkpoint_paths):
            vid_acc, acc, loss, confusion_mat, _, _ = single_shot(
                checkpoint_path + pth_path, epoch+1, len(checkpoint_paths)
            )
            vid_accs.append(vid_acc)
            accs.append(acc)
            losses.append(loss)
            confusion_mats.append(confusion_mat)
        
        result = {
            'test_result': {'vid_acc': vid_accs, 'acc': accs, 'out_loss': losses, 'confusion_mat': confusion_mats},
            'train_result': get_train_result(checkpoint_path + checkpoint_paths[-1]),
        }
        with open(self.config.test['resultDir'] + self.config.basic['tryID'] + '_result.json', 'w') as f:
            json.dump(result, f)
        if self.config.test['onlyEvalLastN'] is None:
            PlotResult(
                result,
                result_path=self.config.test['resultDir'] + self.config.basic['tryID'] + '.png'
            )

        return vid_accs, accs, losses