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