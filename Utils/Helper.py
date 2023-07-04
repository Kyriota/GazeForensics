import torch
import gc
from Config import Config
from Core.TrainManager import TrainManager
from Core.EvaluateManager import EvaluateManager



class ClearCache:
    
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()



def Run(config:Config):

    def PrintKeyConfig():
        space = 24
        print('Key config:\n')
        print(
            ' - Train DS:'.ljust(space) + \
            '{:.1%}'.format(config.prep['util_percent']),
            config.basic['train_DS_name']
        )
        print(
            ' - Test DS:'.ljust(space) + \
            '{:.1%}'.format(config.test['util_percent']),
            config.basic['test_DS_name']
        )
        print(
            ' - Loss Type:'.ljust(space) + config.loss['loss_func'],
                '+ Gaze * {:.2f}'.format(config.loss['gaze_weight']) if config.loss['gaze_weight'] > 0 else '',
                '+ Bonus * {:.2f}'.format(config.loss['bonus_weight']) if config.loss['bonus_weight'] > 0 else ''
        )
        if config.loss['loss_func'] == 'custom':
            print(' ' * space, end='')
            print(
                'FN_w:     {:.2f}'.format(config.loss['FN_w']),
                'FN_bound: {:.2f}'.format(config.loss['FN_bound']),
            )
        print(
            ' - Model Structure:'.ljust(space) + 'Basic', '+ Leaky * {}'.format(config.model['leaky']) if config.model['leaky'] > 0 else ''
        )


    print('-' * 30)
    print("\n >> Starting a new run ...\n")
    PrintKeyConfig()
    print('-' * 30)

    return_state = 0

    if config.train['enable']:

        with ClearCache():
            trainManager = TrainManager(config)
            return_state = trainManager.train()
            del trainManager
        
        gc.collect() 

    if config.test['enable'] and return_state == 0:

        with ClearCache():
            evaluateManager = EvaluateManager(config)
            evaluateManager.evaluate(
                config.basic['checkpointDir'] + config.basic['tryID'] + '/',
                show_progress=True,
                show_confusion_mat=True,
            )
            del evaluateManager

        gc.collect() 