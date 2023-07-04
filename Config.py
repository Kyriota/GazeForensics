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
            'tryID': 'Standard', # If None, tryID will be set to the current time
        }
        params_model = { # Parameters that controls model
            'checkpoint': None, # None stands for init a new model, otherwise load the model from the path
            'seed': 64, # This seed if for initializing the model
            'leaky': 0, # How many dimensions to expand on gaze embedding
            'emb_dim': 512, # Dimension of gaze embedding
            'head_num': 8, # Number of heads in multi-head attention
            'dim_per_head': 64, # Dimension per head in multi-head attention
            'comp_dim': 32, # Dimension of the compressed feature
        }
        params_opti = { # Parameters that controls optimizing, loss, scheduling.
            'lr': 1e-5,
            'weight_decay': 1e-5,
            'step_size': 1,
            'gamma': 0.95,
        }
        params_loss = { # Parameters that controls loss
            'loss_func': 'standard', # Either 'standard' or 'custom'
            'bonus_weight': 0,
            'FN_w': 1, # False negative weight in custom loss
            'FN_bound': 0.6, # False negative boundary in custom loss
            'gaze_weight': 0,
            #   'gaze_weight' decides whether to use gaze info or not
            #   because final_loss = output_BCE_loss + gaze_weight * gaze_MSE_loss
            #   if gaze_weight is 0, the model will not use gaze info
        }
        params_prep = { # Parameters that controls preprocess
            'thread_num': 4, # Note that different thread_num will lead to different result in dataset clipping
            'util_percent': 0.1, # utilization percentage of the dataset
            'tempDir': rootDir + 'temp/', # 'tempDir' is the directory to store clipped train dataset
            'prep_every_epoch': True,
            #   True or False
            #   Due to the randomness of the dataset clipping process,
            #   preprocess the dataset every epoch may alleviate overfitting
            'save_dataset': True, # True or False, this is useful when trying to find the best hyperparameters
            'transform': standard_transform,
            'trainCateDictDir': rootDir + 'clip_info/' + train_DS_name + '_vid_category_dict.json',
        }
        params_test = { # Parameters that controls test process
            'enable': True, # True or False
            'level': 'clip',
            #   Either 'clip' or 'video'
            #   'clip':
            #       Gen test DS by randomly select clips.
            #       If 'util_percent' < 1,
            #       will try to make dataset contains equal amount of clips from all videos.
            #   'video':
            #       Gen test DS by randomly select videos.
            #       If 'util_percent' < 1,
            #       some videos in the DS will not be used at all.
            'batch_size': 16,
            'num_workers': 4,
            'util_percent': 0.25, # utilization percentage of the dataset
            'testClipsDir': rootDir + 'clipped_videos/' + test_DS_name + '_clip/',
            'resultDir': 'Results/',
        }
        params_train = { # Parameters that controls train process
            'enable': True, # True or False
            'num_epochs': 10,
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