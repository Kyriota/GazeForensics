from Utils.Visualization import *
from Utils.Visualization import ProgressBar
from Utils.FileOperation import *

import threading
import json
import cv2
import numpy as np
from time import sleep
import math
# from PIL import Image



def ClipTrainDatasetParallel(
        json_path,
        output_path,
        output_lock,
        thread_num=4,
        util_percent=1.0,
        epoch=0,
    ):
    '''
    Fire several sub threads to clip the train dataset into clips.
    Parameters:
        json_path: Path to the json file describing the dataset.
        output_path: A temporary directory to store the clipped dataset.
        thread_num: Number of sub threads to fire.
        util_percent: Percentage of the dataset to be used.
    '''
    # Read json file
    with open(json_path, 'r') as f:
        vid_category_dict = json.load(f)

    random_seed = int(util_percent * 1e6) + epoch * thread_num
    
    if util_percent < 1.0:
        half_total_len = 0
        for i in vid_category_dict['real_train']:
            half_total_len += i['split_num']
        residual = half_total_len - int(half_total_len * util_percent)
        
        # Shuffle the fake_train and real_train
        np.random.seed(random_seed)
        np.random.shuffle(vid_category_dict['fake_train'])
        np.random.shuffle(vid_category_dict['real_train'])

        warned = False
        for _ in range(residual):
            # Find the video with the max split_num and reduce it by 1
            # If split_num comes to 0, remove the video from the list
            for key in ['fake_train', 'real_train']:
                max_element = max(vid_category_dict[key], key=lambda x: x['split_num'])
                max_element['split_num'] -= 1
                if max_element['split_num'] == 0:
                    if not warned:
                        print('>>> Removing videos from dataset to match util_percent, consider increasing util_percent')
                        warned = True
                    vid_category_dict[key].remove(max_element)
    
    # Split the dataset into several parts
    vid_category_dict_list = []
    for i in range(thread_num):
        vid_category_dict_list.append({
            'len_per_vid': vid_category_dict['len_per_vid'],
            'fake_train': vid_category_dict['fake_train'][i::thread_num],
            'real_train': vid_category_dict['real_train'][i::thread_num],
        })
    
    # Generate a list of random numbers for each thread.
    # This is for reproducibility due to np.random.seed() is not thread safe.
    total_len = 0
    for i in vid_category_dict['fake_train']:
        total_len += i['split_num']
    for i in vid_category_dict['real_train']:
        total_len += i['split_num']
    total_len = math.ceil(total_len / thread_num) * 3 # make random_num_list longer to prevent index out of range
    random_num_list = [[np.random.randint(-1e6, 1e6) for j in range(total_len)] for i in range(thread_num)]

    # Create sub threads
    sub_threads = []
    for i in range(thread_num):
        sub_threads.append(threading.Thread(
            target=ClipTrainDataset,
            args=(
                vid_category_dict_list[i],
                output_path,
                random_num_list[i]
            )
        ))
    rm(output_path, r=True)
    mkdir(output_path)
    
    # Start sub threads
    output_lock.acquire()
    for i in range(thread_num):
        sub_threads[i].start()
        sleep(0.1)
    output_lock.release()

    # Wait for sub threads to finish
    for i in range(thread_num):
        sub_threads[i].join()
    
    # print('>>> Clipping finished.')
    # Save info about this dataset into a json file
    info = {
        'dataset_json_path': json_path,
        'util_percent': util_percent,
        'epoch': epoch,
    }
    with open(output_path + 'info.json', 'w') as f:
        json.dump(info, f)


def ClipTrainDataset(
        vid_category_dict,
        output_path,
        random_num_list=None
    ):

    '''
    Clip the train dataset into clips.
    The clips are stored in a temporary directory.
    Video will be clipped into instructed split_num splits of clips in random step and offset to create variations.
    This function may take a while to finish.
    Parameters:
        vid_category_dict: A dictionary containing the paths of all videos.
        output_path: A temporary directory to store the clipped dataset.
    '''

    def pop_random_num_list(min_num, max_num):
        return random_num_list.pop() % (max_num - min_num) + min_num
    
    get_random_num = pop_random_num_list if random_num_list is not None else np.random.randint
    len_per_vid = vid_category_dict['len_per_vid']

    # rm(output_path, r=True)
    # mkdir(output_path)

    # Create progress bar
    total_len = 0
    for i in vid_category_dict['fake_train']:
        total_len += i['split_num']
    for i in vid_category_dict['real_train']:
        total_len += i['split_num']
    progress_bar = ProgressBar(
        'blue',
        'Clipping',
        total_len
    )

    for vid_category_key in vid_category_dict:
        if vid_category_key == 'len_per_vid' or 'test' in vid_category_key:
            continue
        # print('>>> Processing', vid_category_key)
        mkdir(output_path + vid_category_key)
        for i, clipInfo in enumerate(vid_category_dict[vid_category_key]):
            # print(
            #     'Processing', clipInfo['path'], '\t',
            #     i+1, '\t/\t', len(vid_category_dict[vid_category_key]),
            #     end='                \r'
            # )
            vid_path = clipInfo['path']
            split_num = clipInfo['split_num']
            vid_len = clipInfo['vid_len']
            excess_len = vid_len - len_per_vid

            frames_list = [[] for i in range(split_num)]
            vidcap = cv2.VideoCapture(vid_path)
            for i in range(split_num):
                random_extend = get_random_num(0, excess_len)
                random_offset = get_random_num(0, excess_len-random_extend)
                # print('>>> Random offset:', random_offset, 'Random extend:', random_extend)
                sample_step = (vid_len-excess_len+random_extend) / len_per_vid
                for j in range(len_per_vid):
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, random_offset + int(j * sample_step))
                    success, image = vidcap.read()
                    if success:
                        frames_list[i].append(image)
                    else:
                        # print('\n>>> ERROR: Failed to read the video\n\t', vid_path)
                        return
            vidcap.release()

            '''
            # Display frames for debugging using PIL.Image.show()
            for i in range(split_num):
                print('Split', i)
                for frame in frames_list[i]:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frame.show()
            '''

            # Save the clipped video
            for i in range(split_num):
                vid_name = vid_path.split('/')
                vid_name = vid_name[-2] + '_' + vid_name[-1].split('.')[0] + '_' + str(i) + '.mp4'
                new_vid_path = output_path + vid_category_key + '/' + vid_name
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_size = (frames_list[i][0].shape[1], frames_list[i][0].shape[0])
                out = cv2.VideoWriter(new_vid_path, fourcc, 30.0, out_size)
                for frame in frames_list[i]:
                    out.write(frame)
                out.release()
            
            # Update progress bar
            progress_bar.Update(
                ' * Now processing ' + vid_path + \
                '<br>' + ' * Saving to ' + output_path + vid_category_key + '/' + vid_name,
                increment=split_num
            )

        # print('\n')



def ClipTestDataset(
        json_path,
        output_path
    ):
    '''
    Clip test dataset into clips.
    Videos will be clipped into instructed split_num splits of clips in fixed step and offset.
    The chosen step and offset aim to cover the whole video as complete as possible.
    Which means one footage may be sampled serval times in different steps and offsets.
    This is to ensure the test dataset is as complete as possible so that the model can be evaluated more accurately.
    Parameters:
        json_path: Path to the json file describing the dataset.
        output_path: A temporary directory to store the clipped dataset.
    '''
    # Read the json file
    with open(json_path, 'r') as f:
        vid_category_dict = json.load(f)
    len_per_vid = vid_category_dict['len_per_vid']

    rm(output_path, r=True)
    mkdir(output_path)

    np.random.seed(0)

    for vid_category_key in vid_category_dict:
        if vid_category_key == 'len_per_vid' or 'train' in vid_category_key:
            continue
        print('>>> Processing', vid_category_key)
        mkdir(output_path + vid_category_key)
        for i, clipInfo in enumerate(vid_category_dict[vid_category_key]):
            print(
                'Processing', clipInfo['path'], '\t',
                i+1, '\t/\t', len(vid_category_dict[vid_category_key]),
                end='                \r'
            )
            vid_path = clipInfo['path']
            split_num = clipInfo['split_num']
            vid_len = clipInfo['vid_len']

            frames_list = []
            vidcap = cv2.VideoCapture(vid_path)
            if not vidcap.isOpened():
                print('\n>>> ERROR: Failed to open the video\n\t', vid_path)
                return
            while True:
                success, image = vidcap.read()
                if success:
                    frames_list.append(image)
                else:
                    break
            vidcap.release()

            serial_num = 0
            last_vid_num = None
            for step in range(split_num, 0, -1):
                if step <= 3 and split_num != 1 and serial_num >= 5:
                    break
                vid_num = vid_len // (step * len_per_vid)
                if last_vid_num is not None and last_vid_num+2 >= vid_num:
                    continue
                last_vid_num = vid_num
                redundance = vid_len % (step * len_per_vid)
                random_offset = np.random.randint(0, redundance) if redundance else 0
                for j in range(vid_num):
                    temp_frames = []
                    for k in range(len_per_vid):
                        temp_frames.append(frames_list[(j*len_per_vid+k)*step+random_offset])
                        
                    '''
                    # Display frames for debugging using PIL.Image.show()
                    print('\nSerial', serial_num)
                    for frame in temp_frames:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = Image.fromarray(frame)
                        frame.show()
                    '''
                    
                    vid_name = vid_path.split('/')
                    vid_name = vid_name[-2] + '_' + vid_name[-1].split('.')[0] + '_' + str(serial_num) + '.mp4'
                    new_vid_path = output_path + vid_category_key + '/' + vid_name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_size = (temp_frames[0].shape[1], temp_frames[0].shape[0])
                    out = cv2.VideoWriter(new_vid_path, fourcc, 30.0, out_size)
                    for frame in temp_frames:
                        out.write(frame)
                    out.release()
                    serial_num += 1

        print('\n')