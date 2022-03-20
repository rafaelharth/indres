"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

The main work of this module is done here. Creates the triple rows (images with masks, images without masks, just masks) for each neuron,
  where images are sampled uniformly out the set of images for which the neuron and label mask both have nonzero coverage.
"""
import os
import sys
import shutil
import numpy as np
from PIL import Image
from skimage.transform import resize

import S
import formulas.parser as Parser
import formulas.utils as FU
import settings
import visualization.html_common as HTMLCommon
import score_calculator as ScoreCalculator
import loaders.metadata_loader as MetaData
import visualization.html_summary as HTMLSummary
import loaders.mask_loader as MaskLoader
import util.util as Util

MG = 0.25
BLUE_TINT = np.array([-MG * 255, -MG * 255, MG * 255])
RED_TINT = np.array([MG * 255, -MG * 255, -MG * 255])



def create(R, inspected_neurons, image_width=130):
    def add_colored_masks(img_, label_mask_, neuron_mask_):
        img_ = img_.astype(np.int64)

        label_mask_ = label_mask_[:, :, np.newaxis] * BLUE_TINT
        label_mask_ = np.round(label_mask_).astype(np.int64)
        img_ += label_mask_

        neuron_mask_ = neuron_mask_[:, :, np.newaxis] * RED_TINT
        neuron_mask_ = np.round(neuron_mask_).astype(np.int64)
        img_ += neuron_mask_

        img_ = np.clip(img_, 0, 255).astype(np.uint8)

        return img_

    def map_image_i_to_index_in_original_array(dense_index):
        hits_in_original_array = 0

        for i, image_bool in enumerate(MaskLoader.image_bitmasks[neuron_i]):
            if image_bool:
                hits_in_original_array += 1
                if hits_in_original_array == dense_index:
                    break

        return i


    images_name = f'images_{min(settings.NEURONS)}_{max(settings.NEURONS)}'

    html = []
    neuron_i = int(R["neuron"])
    if settings.EASY_MODE:
        MaskLoader.store_easy_masks(neuron_i)
    threshold = float(R["threshold"]) if settings.DYNAMIC_THRESHOLDS else ScoreCalculator.g["static_thresholds"][neuron_i]




    formula = Parser.parse(R["formula"])


    row1fns = [f"{images_name}/{neuron_i:04d}-{i}.jpg" for i in range(settings.TOP_N)]
    row2fns = [f"{images_name}/{neuron_i:04d}-maskimg-{i}.jpg" for i in range(settings.TOP_N)]
    row3fns = [f"{images_name}/{neuron_i:04d}-masksource-{i}.jpg" for i in range(settings.TOP_N)]

    coverage = lambda mask: mask.sum() / (S.n_images * S.mask_shape[0] * S.mask_shape[1])
    label_masks = FU.compute_composite_mask(formula, neuron_i=(neuron_i if settings.EASY_MODE else None))
    neuron_masks = inspected_neurons[neuron_i] > threshold



    score = HTMLSummary.scores[neuron_i]
    iou = HTMLSummary.ious[neuron_i]

    graytext = " lowscore" if score < settings.SCORE_THRESHOLD else ""
    html.append('<div class="unit%s">' % graytext
        + f"<div class='unitlabel'>{R['formula']}</div>"
        + '<div class="info">'
        + '<span class="unitnum">neuron #%d || </span> ' % neuron_i
        + '<span class="iou">Score: %.3f / IoU: %.3f</span>' % (score, iou)
        + f'<span class="n">Coverage (neuron / label): {coverage(neuron_masks):.2%} / {coverage(label_masks):.2%} </span>'
        + '<span class="n">Threshold: %.2f </span> ' % threshold
        + "</div>"
        + '<p class="midrule">Randomly chosen images where (<span class="bluespan">feature</span> / <span class="redspan">unit</span> mask)'
          ' both have nonzero coverage.</p>')



    label_tallies = label_masks.sum((1, 2))
    neuron_tallies = (inspected_neurons[neuron_i] > threshold).sum((1, 2))

    indices_with_nonzero_neuron_activations = np.where(neuron_tallies > threshold)
    indices_with_nonzero_label_activations = np.where(label_tallies > 0)[0]

    Is = list(np.intersect1d(indices_with_nonzero_neuron_activations, indices_with_nonzero_label_activations))
    try:
        random_subset_of_indices = Util.sample_with_duplicates_if_necessary(Is, settings.TOP_N)
    except ValueError:
        print(label_tallies.shape, "", neuron_tallies.shape)
        sys.exit(3)


    label_masks_resized = [None] * len(random_subset_of_indices)
    neuron_masks_resized = [None] * len(random_subset_of_indices)

    ###########
    # Row I   #
    ###########
    html.append('<div class="thumbcrop">')
    for picture_i, random_image_i in enumerate(random_subset_of_indices):
        if settings.EASY_MODE:
            filename = MetaData.filename(map_image_i_to_index_in_original_array(random_image_i))
        else:
            filename = MetaData.filename(random_image_i)

        img = np.array(Image.open(filename))

        label_mask = label_masks[random_image_i]  # shape=(7,7)
        neuron_activations = inspected_neurons[neuron_i][random_image_i]
        neuron_mask = neuron_activations > threshold # shape=(7,7)

        score = settings.SCORE_FUNCTION_REPORT(np.array([neuron_mask]), np.array([label_mask]))
        score = ScoreCalculator.apply_complexity_penalty(score, len(formula))
        iou = ScoreCalculator.iou(np.array([neuron_mask]), np.array([label_mask]))

        label_masks_resized[picture_i] = resize(label_mask, img.shape[:2], order=0)
        neuron_masks_resized[picture_i] = resize(neuron_mask, img.shape[:2], order=0)

        # Save image in the images/ folder
        img_masked = add_colored_masks(img, label_masks_resized[picture_i], neuron_masks_resized[picture_i])
        Image.fromarray(img_masked).save(os.path.join(settings.OUTPUT_FOLDER, row1fns[picture_i]))

        img_html = f'<img loading="eager" src="{row1fns[picture_i]}" height="{image_width}">'
        html.append(HTMLCommon.wrap_image(img_html, infos=[f"{score:.3f} / {iou:.3f}"]))
    html.append("</div>")


    ###########
    # Row II  #
    ###########
    html.append('<div class="thumbcrop">')
    for picture_i, random_image_i in enumerate(random_subset_of_indices):
        if settings.EASY_MODE:
            img_filename_in_dataset = MetaData.filename(map_image_i_to_index_in_original_array(random_image_i))
        else:
            img_filename_in_dataset = MetaData.filename(random_image_i)

        img_filename_new = row2fns[picture_i]

        # Copy over image; replace old one
        # This is probably not necessary seeing as I just use the original images here
        shutil.copy(img_filename_in_dataset, os.path.join(settings.OUTPUT_FOLDER, img_filename_new))

        img_html = f'<img loading="eager" src="{img_filename_new}" height="{image_width}">'
        html.append(HTMLCommon.wrap_image(img_html, infos=[""]))
    html.append("</div>")


    ###########
    # Row III #
    ###########
    html.append('<div class="thumbcrop">')
    for picture_i, random_image_i in enumerate(random_subset_of_indices):
        if settings.EASY_MODE:
            filename = MetaData.filename(map_image_i_to_index_in_original_array(random_image_i))
        else:
            filename = MetaData.filename(random_image_i)

        dummy_img = np.ones_like(Image.open(filename)) * 256
        img_masked = add_colored_masks(dummy_img, label_masks_resized[picture_i], neuron_masks_resized[picture_i])

        # Save the mask image
        Image.fromarray(img_masked).save(os.path.join(settings.OUTPUT_FOLDER, row3fns[picture_i]))


        img_html = f'<img loading="eager" src="{row3fns[picture_i]}" height="{image_width}">'
        html.append(HTMLCommon.wrap_image(img_html))
    html.append("</div>")

    html.append("</div>")

    if settings.EASY_MODE:
        MaskLoader.delete_easy_masks(neuron_i)

    return "".join(html)