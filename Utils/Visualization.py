import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from tabulate import tabulate



def NormalizeImage(img, max=255):
    img = img.astype(np.float32)
    img -= img.min()
    img *= float(max) / img.max()
    return img.astype(np.uint8)



def smooth_data(data, slice_num):
    # smooth data by averaging
    # len(smoothed_data) = slice_num
    if len(data) < slice_num:
        return data
    smoothed_data = []
    slice_len = len(data) // slice_num
    for i in range(slice_num):
        smoothed_data.append(np.mean(data[i * slice_len : (i + 1) * slice_len]))
    return smoothed_data



def sample_data(data, sample_num=8):
    sampled_data = []
    for i in range(sample_num):
        temp = data[i][0].cpu().detach().numpy()
        temp = np.transpose(temp, (1, 2, 0))
        temp = NormalizeImage(temp)
        sampled_data.append(temp)
    return np.array(sampled_data)



def show_decoder_samples(original, decoded, title='Decoder Samples'):
    num_rows = 4
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3, 3))
    # plt.rcParams['font.size'] = 8
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    for i in range(num_rows // 2):
        for j in range(num_cols):
            index = i * num_cols + j
            axes[i*2, j].imshow(original[index])
            axes[i*2, j].axis('off')
            axes[i*2 + 1, j].imshow(decoded[index])
            axes[i*2 + 1, j].axis('off')
    
    # fig.suptitle(title)
    print('>> ' + title + ':')
    plt.show()



class ProgressBar:
    def __init__(self, color, description, total_len):
        self.progress_bar = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            description=description,
            bar_style='success',
            orientation='horizontal',
            style={'bar_color': color},
        )
        self.info_text = widgets.HTML('')
        self.layout = widgets.VBox([self.progress_bar, self.info_text])
        self.total_len = total_len
        self.progress = 0
        display(self.layout)
    

    def Update(self, info_text_value='', increment=1):
        self.progress += increment
        self.progress_bar.value = self.progress / self.total_len * 100
        self.info_text.value = info_text_value
    


def PlotHistory(history, use_gaze, slice_num=100, global_size=1):
    # Plot training history
    #  - if use_gaze is False, only 2 subplots needed to show out_loss and acc
    #  - if use_gaze is True, 4 subplots needed to show out_loss, acc, total_loss and gaze_loss

    figsize = (12 * global_size, 8 * global_size)
    plt.rcParams['font.size'] = 12 * global_size

    # set up figure
    fig = plt.figure(figsize=figsize)
    axes = [fig.add_subplot(2, 2, i) for i in range(1, 5 if use_gaze else 3)]
    axes[0].set_ylabel('Out Loss')
    axes[1].set_ylabel('Accuracy')
    if use_gaze:
        axes[2].set_ylabel('Total Loss')
        axes[3].set_ylabel('Gaze Loss')

    # smooth data
    smoothed_history = {}
    for key in history.keys():
        if type(history[key]) == list:
            smoothed_history[key] = smooth_data(history[key], slice_num)
    
    # plot
    axes[0].plot(smoothed_history['out_loss'], label='train')
    axes[1].plot(smoothed_history['acc'], label='train')
    if use_gaze:
        axes[2].plot(smoothed_history['total_loss'], label='train')
        axes[3].plot(smoothed_history['gaze_loss'], label='train')
    
    for i in range(4 if use_gaze else 2):
        axes[i].legend()
        axes[i].grid(True)
    
    plt.show()



def PlotResult(result, result_path='result.png', global_size=1):

    eval_result, train_result = result['test_result'], result['train_result']
    figsize = (12 * global_size, 4 * global_size)
    plt.rcParams['font.size'] = 12 * global_size

    # set up figure
    fig = plt.figure(figsize=figsize)
    axes = [fig.add_subplot(1, 2, i) for i in range(1, 3)]
    axes[0].set_ylabel('Out Loss')
    axes[1].set_ylabel('Accuracy')
    
    # plot
    axes[0].plot(eval_result['out_loss'], 'o-', label='eval')
    axes[0].plot(train_result['out_loss'], 'o-', label='train')
    axes[1].plot(eval_result['acc'], 'o-', label='eval')
    axes[1].plot(train_result['acc'], 'o-', label='train')

    for i in range(2):
        axes[i].legend()
        axes[i].grid(True)
        axes[i].yaxis.set_major_locator(plt.MultipleLocator(0.1))
        axes[i].yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    
    fig.savefig(result_path)
    plt.show()



def PrintBinConfusionMat(prediction, ground_truth):
    # Use tabulate to print confusion matrix
    #  - prediction and ground_truth are both 1D array of bool

    # calculate confusion matrix
    tp = np.sum(np.logical_and(prediction == True, ground_truth == True))
    tn = np.sum(np.logical_and(prediction == False, ground_truth == False))
    fp = np.sum(np.logical_and(prediction == True, ground_truth == False))
    fn = np.sum(np.logical_and(prediction == False, ground_truth == True))
    confusion_mat = [
        ['',       'Actual T', 'Actual F'],
        ['Pred T',  tp,         fp       ],
        ['Pred F',  fn,         tn       ]
    ]
    confusion_mat = tabulate(confusion_mat, tablefmt='orgtbl')
    print(' > Confusion Matrix:')
    print(confusion_mat)
    print(' > Total', len(prediction), 'samples')
    return confusion_mat