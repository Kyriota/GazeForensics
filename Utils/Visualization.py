from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from tabulate import tabulate
import cv2
import seaborn as sns
import torch
from PIL import Image



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
    


def PlotHistory(history, use_gaze, slice_num=100, global_size=1):
    # Plot training history
    #  - if use_gaze is False, only 2 subplots needed to show out_loss and acc
    #  - if use_gaze is True, 4 subplots needed to show out_loss, acc, total_loss and gaze_loss

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
    axes[1].plot(eval_result['acc'], 'o-', label='eval-seq')
    axes[1].plot(train_result['acc'], 'o-', label='train')
    axes[1].plot(eval_result['vid_acc'], 'o-', label='eval-vid')

    for i in range(2):
        axes[i].legend()
        axes[i].grid(True)
        axes[i].yaxis.set_major_locator(plt.MultipleLocator(0.1))
        axes[i].yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    
    fig.savefig(result_path)
    plt.show()



def GetBinConfusionMat(prediction, ground_truth, verbose=True):
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
    if verbose:
        print(' > Confusion Matrix:')
        print(confusion_mat)
        print(' > Total', len(prediction), 'samples')
    return confusion_mat



def ShowCAM(cam, model, data_path, transform, show=True, wanted_result_type=None):
    model.eval()
    model.cuda()
    # Load video frames from data_path to `frames` using cv2
    frames = []
    vidcap = cv2.VideoCapture(data_path)
    for i in range(14):
        success, frame = vidcap.read()
        if not success:
            break
        frames.append(np.array(Image.fromarray(frame).resize((224,224))))
    vidcap.release()
    # Transform frames to tensor
    input_tensor = torch.FloatTensor(14, 3, 224, 224)
    for i in range(14):
        input_tensor[i] = transform(Image.fromarray(frames[i]))
    input_tensor = torch.unsqueeze(input_tensor, 0).cuda()
    # Get Result
    prediction = model(input_tensor)
    print(' > Prediction:', prediction.detach().cpu().numpy()[0])
    if wanted_result_type is not None:
        # prediction_type is 'TN', 'TP', 'FN', 'FP', depending on the result and whether data_path have 'real' in its name
        prediction_type = prediction.detach().cpu().numpy()[0][1] > 0.5
        if 'real' in data_path:
            prediction_type = 'TP' if prediction_type else 'FN'
        else:
            prediction_type = 'TN' if not prediction_type else 'FP'
        if prediction_type != wanted_result_type:
            print(' > Skip')
            return None
    # Get CAM
    grayscale_cam = cam(input_tensor)
    results = []
    # Visualization
    for i in range(14):
        visualization = show_cam_on_image(
            frames[i] / 255.0,
            grayscale_cam[i],
            use_rgb=True
        )
        visualization = Image.fromarray(visualization)
        if show:
            visualization.show()
        results.append(visualization)
    return results



def ShowOcclusionSensitivity(model, data_path, transform, cube_size=16, step=8, batch_size=16, show=True, oneImg=False):
    assert 224 % cube_size == 0 and 224 % step == 0
    model.eval()
    model.cuda()
    # Load video frames from data_path to `frames` using cv2
    vidcap = cv2.VideoCapture(data_path)
    frames = []
    for i in range(14):
        success, frame = vidcap.read()
        frames.append(frame)
    vidcap.release()
    # Transform frames to tensor
    input_tensor = torch.FloatTensor(14, 3, 224, 224)
    for i in range(14):
        input_tensor[i] = transform(Image.fromarray(frames[i]))
    input_tensor = torch.unsqueeze(input_tensor, 0).cuda()
    # Get Result
    prediction = model(input_tensor)
    input_tensor = input_tensor.detach().cpu()
    del input_tensor
    print(' > Prediction:', prediction.detach().cpu().numpy()[0])
    occlusion_preds = []
    # Initialize an empty heatmap to visualize occlusion sensitivity
    heatmap = np.zeros((224, 224))
    # Fill the occlusion block with random noise
    end_value = 224 // step
    start_value = -(cube_size // step - 1)
    batch_patch = torch.FloatTensor(batch_size, 14, 3, 224, 224).cuda()
    coordinate_rec = []
    batch_cnt = 0
    for i in range(start_value, end_value):
        for j in range(start_value, end_value):

            # Define the occlusion block coordinates
            occlusion_y_start = j * step
            occlusion_x_start = i * step
            occlusion_x_end = occlusion_x_start + cube_size
            occlusion_y_end = occlusion_y_start + cube_size
            occlusion_y_start = 0 if occlusion_y_start < 0 else occlusion_y_start
            occlusion_x_start = 0 if occlusion_x_start < 0 else occlusion_x_start
            occlusion_x_end = 224 if occlusion_x_end > 224 else occlusion_x_end
            occlusion_y_end = 224 if occlusion_y_end > 224 else occlusion_y_end

            # Apply occlusion by replacing the block
            occluded_input = torch.FloatTensor(14, 3, 224, 224)
            occlusion_cube = np.ones((occlusion_x_end-occlusion_x_start, occlusion_y_end-occlusion_y_start, 3)) * 127
            if oneImg:
                original_img = np.array(Image.fromarray(frames[7]).resize((224, 224)))
                original_img[occlusion_x_start:occlusion_x_end, occlusion_y_start:occlusion_y_end] = occlusion_cube
                for k in range(14):
                    occluded_input[k] = transform(Image.fromarray(original_img))
            else:
                for k in range(14):
                    original_img = np.array(Image.fromarray(frames[k]).resize((224, 224)))
                    original_img[occlusion_x_start:occlusion_x_end, occlusion_y_start:occlusion_y_end] = occlusion_cube
                    occluded_input[k] = transform(Image.fromarray(original_img))

            # # Debug code: Save iamges
            # save_img = NormalizeImage(occluded_input[0].permute(1,2,0).detach().cpu().numpy())
            # Image.fromarray(save_img[:, :, ::-1]).save('Debug/' + str(len(occlusion_preds)) + '.jpg')
            
            batch_patch[batch_cnt] = occluded_input
            coordinate_rec.append({
                'occlusion_y_start': occlusion_y_start,
                'occlusion_x_start': occlusion_x_start,
                'occlusion_x_end': occlusion_x_end,
                'occlusion_y_end': occlusion_y_end,
            })
            batch_cnt += 1

            if batch_cnt != batch_size and i + j < end_value - 2:
                continue

            batch_cnt = 0

            # Make a prediction for the occluded input
            batch_occluded_pred = model(batch_patch.cuda())

            # Store the prediction result for this occlusion configuration
            for CR, occluded_pred in zip(coordinate_rec, batch_occluded_pred):
                occlusion_preds.append(occluded_pred.detach().cpu().numpy()[0])

                # Calculate the sensitivity score based on the difference between original prediction and occluded prediction
                sensitivity_score = np.abs(prediction.detach().cpu().numpy()[0][0] - occluded_pred.detach().cpu().numpy()[0])

                # Update the corresponding region in the heatmap with the sensitivity score
                heatmap[CR['occlusion_x_start']:CR['occlusion_x_end'], CR['occlusion_y_start']:CR['occlusion_y_end']] += sensitivity_score

            coordinate_rec = []

            print(len(occlusion_preds), 'of', (end_value - start_value) ** 2, end='       \r')

    # Convert the list of predictions to a NumPy array
    occlusion_preds = np.array(occlusion_preds)

    # Normalize the heatmap for visualization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Blend the heatmap with the original frame
    mean_frame = np.mean(frames, axis=0).astype(np.uint8)
    mean_frame = np.array(Image.fromarray(mean_frame).resize((224, 224)))
    heatmap_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlaid_image = cv2.addWeighted(mean_frame, 0.7, heatmap_img, 0.3, 0)

    # Display the overlaid image
    final_img = Image.fromarray(cv2.cvtColor(overlaid_image, cv2.COLOR_BGR2RGB))
    if show:
        final_img.show()

    return final_img, heatmap