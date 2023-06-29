import torchvision.transforms as transforms


rootDir = '/home/kyr/GazeForensicsData/'


class Config:
    def __init__(self):

        train_DS_name = 'WDF'
        test_DS_name = 'WDF'

        standard_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        params_basic = { # Fundamental parameters
            'rootDir': rootDir,
            'train_DS_name': train_DS_name,
            'test_DS_name': test_DS_name,
            'checkpointDir': 'Checkpoints/',
            'tryID': None, # 'tryID' is datetime in yymmdd_HHMMSS string
        }
        params_model = { # Parameters that controls model
            'checkpoint': None, # None stands for init a new model, otherwise load the model from the path
            'seed': 64, # This seed if for initializing the model
            'leaky': 0, # How many dimensions to expand on gaze embedding
        }
        params_opti = { # Parameters that controls optimizing, loss, scheduling.
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'step_size': 1,
            'gamma': 1,
        }
        params_loss = { # Parameters that controls loss
            'loss_func': 'standard', # Either 'standard' or 'custom'
            'bonus_weight': 0,
            'FN_w': 4, # False negative weight in custom loss
            'FN_bound': 0.4, # False negative boundary in custom loss
            'gaze_weight': 0, # 'gaze_weight' decides whether to use gaze info or not
            #                         because final_loss = output_BCE_loss + gaze_weight * gaze_MSE_loss
            #                         if gaze_weight is 0, the model will not use gaze info
        }
        params_prep = { # Parameters that controls preprocess
            'thread_num': 4, # Note that different thread_num will lead to different result in dataset clipping
            'util_percent': 0.1, # utilization percentage of the dataset
            'tempDir': rootDir + 'temp/', # 'tempDir' is the directory to store clipped train dataset
            'prep_every_epoch': True, # True or False
            'transform': standard_transform,
            'trainCateDictDir': rootDir + 'clip_info/' + train_DS_name + '_vid_category_dict.json',
        }
        params_test = { # Parameters that controls test process
            'batch_size': 16,
            'num_workers': 4,
            'util_percent': 0.1, # utilization percentage of the dataset
            'testClipsDir': rootDir + 'clipped_videos/' + test_DS_name + '_clip/',
        }
        params_train = { # Parameters that controls train process
            'num_epochs': 3,
            'batch_size': 16,
            'num_workers': 4,
            'seed': 64, # This seed if for the random shuffle of the dataset in training
        }


        self.basic = params_basic
        self.model = params_model
        self.opti = params_opti
        self.prep = params_prep
        self.test = params_test
        self.train = params_train
        self.loss = params_loss