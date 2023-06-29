import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from tabulate import tabulate



def NormalizeImage(img, max=255):
    img = img.astype(np.float32)
    img -= img.min()
    img *= float(max) / img.max()
    return img.astype(np.uint8)



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
    


def PlotHistory(history, use_gaze, smooth_window_size=10, global_size=1):
    # Plot training history
    #  - if use_gaze is False, only 2 subplots needed to show out_loss and acc
    #  - if use_gaze is True, 4 subplots needed to show out_loss, acc, total_loss and gaze_loss

    def smooth_data(data, window_size):
        if len(data) < window_size:
            return data
        data = data[:]
        # for _ in range(window_size - 1):
        #     data.insert(0, data[0])
        #     data.append(data[-1])
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    
    figsize = (12 * global_size, 8 * global_size)
    plt.rcParams['font.size'] = 12 * global_size

    # set up figure
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('Out Loss')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Accuracy')
    if use_gaze:
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title('Total Loss')
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title('Gaze Loss')

    # smooth data
    smoothed_history = {}
    for key in history.keys():
        if type(history[key]) == list:
            smoothed_history[key] = smooth_data(history[key], smooth_window_size)
    
    # plot
    ax1.plot(smoothed_history['out_loss'], label='train')
    ax1.legend()
    ax2.plot(smoothed_history['acc'], label='train')
    ax2.legend()
    if use_gaze:
        ax3.plot(smoothed_history['total_loss'], label='train')
        ax3.legend()
        ax4.plot(smoothed_history['gaze_loss'], label='train')
        ax4.legend()
    
    plt.show()



def PlotResult(eval_history, train_history, global_size=1):

    figsize = (12 * global_size, 8 * global_size)
    plt.rcParams['font.size'] = 12 * global_size

    len_per_epoch = train_history['len_per_epoch']
    train_history['out_loss'] = [
        np.mean(
            train_history['out_loss'][i:i+len_per_epoch]
        ) for i in range(0, len(train_history['out_loss']), len_per_epoch)
    ]
    train_history['acc'] = [
        np.mean(
            train_history['acc'][i:i+len_per_epoch]
        ) for i in range(0, len(train_history['acc']), len_per_epoch)
    ]

    # set up figure
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('Out Loss')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Accuracy')
    
    # plot
    ax1.plot(eval_history['out_loss'], label='eval')
    ax1.plot(train_history['out_loss'], label='train')
    ax1.legend()
    ax2.plot(eval_history['acc'], label='eval')
    ax2.plot(train_history['acc'], label='train')
    ax2.legend()
    
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
    print(' > Confusion Matrix:')
    print(tabulate(confusion_mat, tablefmt='orgtbl'))
    print(' > Total', len(prediction), 'samples')