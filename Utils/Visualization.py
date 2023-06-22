import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets



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
    

    def Update(self, info_text_value, increment=1):
        self.progress += increment
        self.progress_bar.value = self.progress / self.total_len * 100
        self.info_text.value = info_text_value
    


def PlotHistory(history, use_gaze, smooth_window_size=10):
    # Plot training history
    #  - if use_gaze is False, only 2 subplots needed to show out_loss and acc
    #  - if use_gaze is True, 4 subplots needed to show out_loss, acc, total_loss and gaze_loss

    def smooth_data(data, window_size):
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    
    # set up figure
    fig = plt.figure(figsize=(12, 8))
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