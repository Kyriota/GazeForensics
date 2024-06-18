import dis
from Config import Config
from Core.TrainManager import TrainManager
from Core.EvaluateManager import EvaluateManager
from Utils.FileOperation import fileWalk

import torch
import gc
import cv2
import numpy as np


class ClearCache:

    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


def Run(config: Config):

    def PrintKeyConfig():
        space = 24
        print('Key config:\n')
        print(
            ' - Train DS:'.ljust(space) +
            '{:.1%}'.format(config.prep.util_percent),
            config.basic.train_DS_name
        )
        print(
            ' - Test DS:'.ljust(space) +
            '{:.1%}'.format(config.test.util_percent),
            config.basic.test_DS_name
        )
        print(
            ' - Loss Type:'.ljust(space) + config.loss.loss_func,
            '+ Gaze * {:.2f}'.format(config.loss.gaze_weight) if config.loss.gaze_weight > 0 else '',
            # '+ Bonus * {:.2f}'.format(config.loss['bonus_weight']) if config.loss['bonus_weight'] > 0 else ''
        )
        if config.loss.loss_func == 'custom':
            print(' ' * space, end='')
            print(
                'FN_w:     {:.2f}'.format(config.loss['FN_w']),
                'FN_bound: {:.2f}'.format(config.loss['FN_bound']),
            )
        print(
            ' - Model Structure:'.ljust(space) + 'Basic', '+ Leaky * {}'.format(config.model.leaky) if config.model.leaky > 0 else ''
        )

    print('-' * 30)
    print("\n >> Starting a new run ...\n")
    PrintKeyConfig()
    print('-' * 30)

    return_state = 0

    if config.train.enable:

        with ClearCache():
            trainManager = TrainManager(config)
            return_state = trainManager.train()
            del trainManager

        gc.collect()

    if config.test.enable and return_state == 0:

        with ClearCache():
            evaluateManager = EvaluateManager(config)
            vid_accs, accs, losses, vid_aucs = evaluateManager.evaluate(
                config.basic.checkpointDir + config.basic.tryID + '/',
                show_progress=True,
                show_confusion_mat=True,
            )

        gc.collect()

    if config.test.enable and return_state == 0:
        best_vid_acc, best_vid_acc_epoch = max(vid_accs), vid_accs.index(max(vid_accs))
        best_vid_auc, best_vid_auc_epoch = max(vid_aucs), vid_aucs.index(max(vid_aucs))
        print('Best vid_acc: {:.4f} at epoch {}, with vid_auc: {:.4f}'.format(
            best_vid_acc, best_vid_acc_epoch + 1, vid_aucs[best_vid_acc_epoch]))
        if best_vid_acc_epoch != best_vid_auc_epoch:
            print('Best vid_auc: {:.4f} at epoch {}, with vid_acc: {:.4f}'.format(
                best_vid_auc, best_vid_auc_epoch + 1, vid_accs[best_vid_auc_epoch]))
        return vid_accs, accs, losses, vid_aucs

    return 0, 0, 0, 0


def GetMeanStd(config: Config, epochs: int = 5):
    ds_name = config.basic.train_DS_name
    ds_paths = [config.basic.rootDir + 'temp/' + ds_name + '_' + '{:.1%}'.format(config.prep.util_percent) + '_epoch' + str(i) + '/' for i in range(epochs)]
    videos_paths = []
    for ds_path in ds_paths:
        temp_paths = fileWalk(ds_path)
        videos_paths.extend([ds_path + file for file in temp_paths if file.endswith('.mp4')])

    print('Calculating mean and std for', ds_name, '...')
    print('Total videos:', len(videos_paths))

    mean = np.zeros(3)
    std = np.zeros(3)
    frame_cnt = 0

    for i, video_path in enumerate(videos_paths):

        if i % 100 == 0 or i == len(videos_paths) - 1:
            print('Calculating Mean For', i + 1, '/', len(videos_paths), '         ', end='\r')

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255  # normalize pixel values

            mean += np.mean(frame, axis=(0, 1))
            frame_cnt += 1

        cap.release()

    print()
    mean /= frame_cnt

    pixel_cnt = 0
    for i, video_path in enumerate(videos_paths):

        if i % 100 == 0 or i == len(videos_paths) - 1:
            print('Calculating Std For', i + 1, '/', len(videos_paths), '         ', end='\r')

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255

            std += np.sum((frame - mean) ** 2, axis=(0, 1))
            pixel_cnt += frame.shape[0] * frame.shape[1]

        cap.release()

    std = np.sqrt(std / pixel_cnt)

    print("\nMean:", mean, "\nStd:", std)

    return mean, std
