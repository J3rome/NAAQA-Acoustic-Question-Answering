from copy import deepcopy

import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
from IPython.core.display import display, Markdown
from IPython.display import Audio

from data_interfaces.transforms import NormalizeInverse
from utils.visualization import show_tagged_scene, get_tagged_scene_table_legend
from utils.notebook.generic import notebook_input_prompt
from runner import custom_game_inference, create_game_for_custom_question
from visualization import one_game_gradcam
from utils.visualization import merge_gradcam_heatmap_with_image, print_top_preds


def do_custom_question_inference(device, model, dataloader, custom_question, scene_id, nb_top_pred=5):
    custom_game = create_game_for_custom_question(dataloader, custom_question[0], scene_id)
    top_preds = custom_game_inference(device, model, custom_game, dataloader, nb_top_pred=nb_top_pred)

    # We decode the question to show the user if there is some <unk>
    print_top_preds(top_preds, dataloader.dataset.tokenizer.decode_question(custom_game['question']),
                    answer=custom_question[1])

    return custom_game, top_preds


def show_gradcam(device, model, dataloader, custom_game, scene_id, guess_id=0, top_preds=None, clear_stats=None,
                 apply_relu=False, target_layers=None):

    if top_preds:
        class_idx = top_preds[guess_id][1]
    else:
        class_idx = None

    heatmaps, confidence = one_game_gradcam(device, model, custom_game, dataloader.collate_fn,
                                            class_idx=class_idx, return_heatmap=True, apply_relu=apply_relu,
                                            target_layers=target_layers)

    #display(get_tagged_scene_table_legend(dataloader, scene_id, colors))

    if clear_stats:
        inverse_norm = NormalizeInverse(clear_stats['mean'],
                                        clear_stats['std'])
        # We copy the game to avoid modifying the original object
        custom_game = inverse_norm(deepcopy(custom_game))

    title = "### GradCam visualization"

    if top_preds is not None:
        title += f" Guess #{guess_id + 1}  ---- [{top_preds[guess_id][0]}]"

    display(Markdown(title))

    nb_layers = len(heatmaps.keys())
    nb_layers_per_fig = 1

    for i, (layer_name, heatmap) in enumerate(heatmaps.items()):
        # if i == 2:
        #    break
        fig_idx = i % nb_layers_per_fig
        if fig_idx == 0:
            fig, axs = plt.subplots(nb_layers_per_fig, 3)

        merged_heatmap = merge_gradcam_heatmap_with_image(heatmap, custom_game['image'])

        line_axs = axs#[fig_idx]

        line_axs[0].set_title(f"[{layer_name}]\nHeatmap")
        line_axs[0].imshow(ToPILImage()(heatmap))
        line_axs[1].set_title("Heatmap Merged\nwith Input Image")
        line_axs[1].imshow(ToPILImage()(merged_heatmap))
        line_axs[2].set_title("Tagged scene")
        fig_ax, colors = show_tagged_scene(dataloader.dataset, custom_game, scene_image=merged_heatmap, fig_ax=(fig, line_axs[2]),
                                          show_legend=False)

    return heatmaps


def show_game_notebook_input(dataloader, game, clear_stats=None, remove_image_padding=False, sound_player=True):
    if clear_stats:
        inverse_norm = NormalizeInverse(clear_stats['mean'],
                                        clear_stats['std'])
        # We copy the game to avoid modifying the original object
        game = inverse_norm(deepcopy(game))

    if sound_player:
        path = f"./{dataloader.dataset.root_folder_path}/audio/train/CLEAR_{dataloader.dataset.set}_{game['scene_id']:06d}.flac"
        display(Audio(path))

    (fig, ax), colors = show_tagged_scene(dataloader.dataset, game, fig_title=f"Scene #{game['scene_id']}",
                                         show_legend=False, remove_padding=remove_image_padding)

    legend = get_tagged_scene_table_legend(dataloader, game['scene_id'], colors)
    display(legend)
    # TODO : Use CSS To move table beside figure
    # legend = get_tagged_scene_table_legend(dataloaders[set_type], scene_id, colors)
    # display(HTML(legend.render()))

    games_per_family = dataloader.dataset.get_random_game_per_family_for_scene(game['scene_id'], return_game=True)
    custom_questions = []

    for i, (family, game) in enumerate(games_per_family.items()):
        if game is None:
            print(f"\nNo question of family {family} for this scene\n")
            continue

        decoded_question = dataloader.dataset.tokenizer.decode_question(game['question'].tolist())
        decoded_question = f"{' '.join(decoded_question.split(' ')[:-1]).capitalize()} ?"
        decoded_answer = dataloader.dataset.tokenizer.decode_answer(game['answer'].item())

        custom_questions.append((decoded_question, decoded_answer))
        question_type_str = f"[{family.capitalize()}]Question ({game['id']}):"
        print(f"{question_type_str: <50}Answer for proposed question: {decoded_answer}")
        notebook_input_prompt(f'custom_question', decoded_question, default_answer=decoded_answer, selected=(i == 0))
        # print(f"Answer for proposed question : {decoded_answer}")

    default_question = custom_questions[0]

    return default_question, custom_questions, legend
