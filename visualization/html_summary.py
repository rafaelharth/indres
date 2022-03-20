"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

A rare remnant from the project's old structure. I wouldn't have made this its own class.
"""


import os
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import settings
import visualization.html_common as HTMLCommon
import visualization.neuron_cards as Cards
import formulas.utils as FU
import score_calculator as ScoreCalculator
import formulas.parser as Parser
import loaders.mask_loader as MaskLoader

ious = {}
scores = {}

def create(map_n_im_2_activations, beam_search_results):
    global ious, scores

    images_name = f'images_{min(settings.NEURONS)}_{max(settings.NEURONS)}'

    print(f"Generating html summary in results_{min(settings.NEURONS)}_{max(settings.NEURONS)}.html")

    html = [HTMLCommon.HTML_PREFIX]

    for R in tqdm.tqdm(beam_search_results, desc="Computing Masks and Scores of formulas"):
        if settings.EASY_MODE:
            MaskLoader.store_easy_masks(R["neuron"])

        threshold = float(R["threshold"]) if settings.DYNAMIC_THRESHOLDS else ScoreCalculator.g["static_thresholds"][R["neuron"]]
        label_masks = FU.compute_composite_mask(Parser.parse(R["formula"]), neuron_i=(int(R["neuron"]) if settings.EASY_MODE else None))
        neuron_masks = map_n_im_2_activations[R["neuron"]] > threshold

        ious[R["neuron"]] = ScoreCalculator.iou(neuron_masks, label_masks)
        scores[R["neuron"]] = settings.SCORE_FUNCTION_REPORT(neuron_masks, label_masks)

        if settings.EASY_MODE:
            MaskLoader.delete_easy_masks(R["neuron"])


    # Create the bar chart summarizing scores
    iou_filename = f"{images_name}/layer4-iou.svg"
    # print(f"The last thing it does is this: {expdir.fn_safe(settings.LAYER_NAME)}")
    scores_ = [scores[r["neuron"]] for r in beam_search_results]
    iou_mean = np.mean(scores_)
    iou_std = np.std(scores_)
    iou_title = f"Scores: ({iou_mean:.3f} +/- {iou_std:.3f})"
    score_histogram(beam_search_results, os.path.join(settings.OUTPUT_FOLDER, iou_filename), title=iou_title)
    html.extend([
            '<div class="histogram">',
            '<img class="img-fluid" src="%s" title="Summary of %s">'
            # % (iou_filename, settings.OUTPUT_FOLDER.split('/')[1], settings.LAYER_NAME),
            % (iou_filename, settings.OUTPUT_FOLDER.split('/')[1]),
            "</div>"])



    html.append('<div class="unitgrid">')

    # Visualize neurons
    for record_i, record in enumerate(tqdm.tqdm(beam_search_results, desc="Visualizing Results")):
        card_html = Cards.create(record, map_n_im_2_activations)
        html.append(card_html)

    html.append("</div>")
    html.extend([HTMLCommon.HTML_SUFFIX])

    with open(os.path.join(settings.OUTPUT_FOLDER, f"{min(settings.NEURONS)}_{max(settings.NEURONS)}.html"), "w") as f:
        f.write("\n".join(html))


def score_histogram(records, filename, title="IoUs"):
    plt.figure()
    # scores_ = [scores[r["neuron"]] for r in records]
    sns.histplot(scores).set_title(title)
    plt.savefig(filename)