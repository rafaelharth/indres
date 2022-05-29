"""
Computes scores for the specific (neuron, formulas) pairs.
"""

import numpy as np

import formulas.utils as FU
import formulas.parser as Parser
import loaders.mask_loader as MaskLoader
import settings
import util.information_printer as PI
import score_calculator as ScoreCalculator

def partial_imroun(map_im_2_neuron_mask, map_im_2_label_mask, indices=None):
    """ Computes the normalized ImRoU_r score, with r = settings.MULTIPLIER_IP. """


    neuron_area_total, label_area_total, intersection_total, random_intersection_total, max_random_intersection_total = 0, 0, 0, 0, 0
    I = map_im_2_label_mask.shape[1] * map_im_2_label_mask.shape[2]
    r = settings.MULTIPLIER_IP



    for i, (neuron_mask, label_mask) in enumerate(zip(map_im_2_neuron_mask, map_im_2_label_mask)):
        if indices is None or i in indices:
            neuron_area_image = neuron_mask.sum()
            label_area_image = label_mask.sum()

            neuron_area_total += neuron_area_image
            label_area_total += label_area_image
            intersection_total += np.logical_and(neuron_mask, label_mask).sum()
            random_intersection_total += neuron_area_image * label_area_image
            max_random_intersection_total += neuron_area_image * neuron_area_image

    union_total = neuron_area_total + label_area_total - intersection_total + 1e-10

    imrou_score = (intersection_total - (r / I) * random_intersection_total) / union_total
    max_imrou_score = (neuron_area_total - (r / I) * max_random_intersection_total) / neuron_area_total


    # return imrou_score / max_imrou_score
    return imrou_score

def compute_scores(map_n_im_2_activations, neuron_i, formula_string, threshold, indices):
    if settings.EASY_MODE:
        MaskLoader.store_easy_masks(neuron_i)




    formula = Parser.parse(formula_string)

    neuron_masks = map_n_im_2_activations[neuron_i] > threshold
    label_masks = FU.compute_composite_mask(formula, neuron_i=(neuron_i if settings.EASY_MODE else None))
    # label_masks = FU.compute_composite_mask(formula)


    score = partial_imroun(neuron_masks, label_masks, None)
    abridged_score = partial_imroun(neuron_masks, label_masks, indices)
    iou = ScoreCalculator.iou(neuron_masks, label_masks)
    # abridged_score = score

    PI.show_simple(f"Neuron {neuron_i} || score = {score:.4f} || iou = {iou},"
                   f" th = {threshold:.3f} (), formulas = {formula.to_str()}.")
    return score, abridged_score


