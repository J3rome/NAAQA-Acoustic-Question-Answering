import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as font_manager
from torchvision.transforms import ToPILImage
import numpy as np
import torch
import pandas as pd

from utils.random import get_random_state, set_random_state
from models.torchsummary import summary     # Custom version of torchsummary to fix bugs with input


def print_model_summary(model, input_image_torch_shape, device="cpu"):
    # Printing summary affects the random state (Raw Vs Pre-Extracted Features).
    # We restore it to ensure reproducibility between input type
    random_state = get_random_state()
    summary(model, [
        ((22,), torch.LongTensor),
        ((1,), torch.LongTensor),
        (input_image_torch_shape, torch.FloatTensor)], device=device)
    set_random_state(random_state)


def get_tagged_scene(dataset, game_or_game_id, scene_image=None, remove_padding=False, show_legend=True, show_fig=False,
                     fig_title=None, fig_ax=None):
    assert dataset.is_raw_img() or scene_image is not None, 'Image to tag must be provided if not in RAW mode'

    if type(game_or_game_id) == int:
        # Game id was supplied
        game = dataset[game_or_game_id]
    else:
        game = game_or_game_id

    if scene_image is not None:
        image = scene_image
    else:
        image = game['image']

    image_padding = game['image_padding'].tolist()
    image_height, image_width = image.shape[1:]

    if remove_padding:
        # Crop image
        image = image[:, :-image_padding[0], :-image_padding[1]]
        image_height, image_width = image.shape[1:]

    # Create figure
    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()#1, figsize=((image_width + 50)/100, (image_height + 150)/100), dpi=100)

    if fig_title is not None:
        fig.suptitle(fig_title)
    ax.imshow(ToPILImage()(image))

    # Retrieve scene informations
    scene = dataset.scenes[game['scene_id']]['definition']
    scene_duration = sum([o['duration'] + o['silence_after'] for o in scene['objects']]) + scene['silence_before']
    time_resolution = int(scene_duration/image_width + 0.5)   # Ms/pixel

    # Generate annotations
    annotations = {}
    rgb_colors = []
    annotation_colormap = plt.cm.get_cmap('hsv', len(scene['objects']))

    current_position = int(scene['silence_before'] / time_resolution)
    for i, sound in enumerate(scene['objects']):
        sound_duration_in_px = int(sound['duration']/time_resolution + 0.5)
        sound_silence_in_px = int(sound['silence_after']/time_resolution + 0.5)
        annotation_color = annotation_colormap(i)
        annotation_rect = patches.Rectangle((current_position, 2), width=sound_duration_in_px,
                                            height=image_height - 4, fill=False, color=annotation_color,
                                            linewidth=1.4)
        key = f"{sound['instrument'].capitalize()}/{sound['brightness']}/{sound['loudness']}/{sound['note']}/{sound['id']}"
        annotations[key] = annotation_rect
        ax.add_patch(annotation_rect)

        current_position += sound_duration_in_px + sound_silence_in_px

        rgb_colors.append(tuple(int(c*255) for c in annotation_color))

    # TODO : Add correct scale to axis (Freq & time)
    if show_legend:
        ax.legend(annotations.values(), annotations.keys(), bbox_to_anchor=(0.5, -0.45), loc='lower center', ncol=2,
                  prop=font_manager.FontProperties(family='sans-serif', size='small'))

    fig.tight_layout()

    if show_fig:
        plt.show()

    return (fig, ax), rgb_colors


def df_col_styler(col_colors=None):
    # Pandas dataframe styler. Each columns will have a color defined by 'col_colors'
    default_style = "text-transform: capitalize;"

    def apply_style(x):
        # copy df to new - original data are not changed
        df = x.copy()

        for i in range(len(df.columns)):
            if col_colors:
                color = f"rgba({col_colors[i][0]},{col_colors[i][1]},{col_colors[i][2]}, 0.6)"
                style = f"{default_style} background-color: {color};"
            else:
                style = default_style
            df[i] = style

        return df

    return apply_style


def get_tagged_scene_table_legend(dataloader, scene_id, col_colors=None):
    sounds = dataloader.dataset.scenes[scene_id]['definition']['objects']

    legend = pd.DataFrame(sounds, columns=['instrument', 'loudness', 'brightness', 'note', 'id']).T

    legend.style.set_table_attributes("style='display:inline'").set_caption('Caption table')

    legend = legend.style.apply(df_col_styler(col_colors), axis=None)
    return legend


def print_top_preds(top_preds, question, answer=None):
    print(f"Question : {question}")
    if answer is not None:
        if answer == top_preds[0][0]:
            print("Correct Answer")
        else:
            print(f"Wrong Answer. Correct answer is : {answer}")

    for i, (ans, class_id, prob) in enumerate(top_preds):
        print("{:>10} {:>25} ---- {}".format(f"Guess {i + 1}:", ans.capitalize(), str(prob)))


def save_graph_to_tensorboard(model, tensorboard, input_image_torch_shape):
    # FIXME : For now we are ignoring TracerWarnings. Not sure the saved graph is 100% accurate...
    import warnings
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

    # FIXME : Test on GPU
    dummy_input = [torch.ones(2, 22, dtype=torch.long),
                   torch.ones(2, 1, dtype=torch.long),
                   torch.ones(2, *input_image_torch_shape, dtype=torch.float)]
    tensorboard['writers']['train'].add_graph(model, dummy_input)

import cv2
def get_gradcam_heatmap(mask):
    # Remove unnecessary dimensions
    while len(mask.shape) > 2:
        mask = mask.squeeze(0)

    mask = mask.detach().cpu().numpy()

    heatmap = (mask * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    return heatmap


def merge_gradcam_heatmap_with_image(heatmap, image):

    while len(image.shape) > 3:
        # Remove batch dimension
        image = image.squeeze(0)

    image = image.detach().cpu()

    merged = heatmap + image

    return merged.div(merged.max())


def visualize_cam(mask, img):
    """ Taken from https://github.com/vickyliin/gradcam_plus_plus-pytorch
    Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    import cv2       # Not an official dependency
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result
