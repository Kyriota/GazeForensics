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

    print('-' * 30)
    print("\n >> Starting a new run ...\n")
    print(
        'Key config:\n\n' + \
        ' - Train DS: {:.1%}'.format(config.prep['util_percent']), config.basic['train_DS_name'], '\n' + \
        ' - Test DS: {:.1%}'.format(config.test['util_percent']), config.basic['test_DS_name'], '\n' + \
        ' - Loss Type:', config.loss['loss_func'], '+ Gaze * {:.2f}'.format(config.loss['gaze_weight']) if config.loss['gaze_weight'] > 0 else '' + \
            ' + Bonus * {:.2f}'.format(config.loss['bonus_weight']) if config.loss['bonus_weight'] > 0 else '', '\n' + \
        ' - Model Structure:', 'Basic', '+ Leaky * {}'.format(config.model['leaky']) if config.model['leaky'] > 0 else '', '\n'
    )
    print('-' * 30)

    with ClearCache():
        trainManager = TrainManager(config)
        trainManager.train()
        del trainManager
    
    gc.collect() 

    with ClearCache():
        evaluateManager = EvaluateManager(config)
        evaluateManager.evaluate(
            config.basic['checkpointDir'] + config.basic['tryID'] + '/',
            show_progress=True,
            show_confusion_mat=True,
        )
        del evaluateManager

    gc.collect() 