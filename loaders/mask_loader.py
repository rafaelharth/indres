"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

Loads and stores label masks.
"""
import os
import PIL
import tqdm
import imageio
import multiprocessing
import numpy as np

import S
import settings
import formulas.formula as F
import formulas.utils as FU
import score_calculator as ScoreCalculator
import loaders.metadata_loader as MetaData

image_quantities = np.array([])
image_bitmasks = np.array([])


thresholds = [128, 120, 110, 100, 140]
all_p_formulas = []
# -----------------------------------------------------------------------------------------------------------------------------------------
# - This loads the ground truth for label masks from the {image_name}_{category_name}.png files. Each such file encodes the mask          -
# -   information for an entire group of possible labels. The purpose of the grouping into categories is such that each group has the     -
# -   property that no two labels within a group can be activate at the same pixel of one image. Thus, a Pixel # in such a png file       -
# -   indicates that one particular label is activate at that position. For example, a pixel in a {image_name}_color.png file indicates   -
# -   that a particular color feature is active at that pixel. Thus, if a category has N labels, then the file holds the same information -
# -   as N binary masks, one for each label. The files are in dataset/.../images/.                                                        -
# -----------------------------------------------------------------------------------------------------------------------------------------
def init(map_n_im_2_activations=None, static_thresholds=None, densify=True):
    global all_p_formulas, image_quantities, image_bitmasks

    def create_masks():
        # Initially, set up the result ordered by images because that's how we process it. Swap it around once we 're done.
        map_im_l_2_bitmask = np.zeros((S.n_images, S.n_labels + 1, *S.mask_shape), dtype=bool)

        if os.path.exists(mask_file_path_regular): # in this case. settings.EXPANSIONS is True (otherwise, we wouldn't be in this method)
            map_pf_im_2_bitmask_ = np.zeros((2 * S.n_labels + 1, S.n_images, *S.mask_shape), dtype=bool)
            map_pf_im_2_bitmask_[0: S.n_labels + 1] = np.load(mask_file_path_regular)

            for i in tqdm.tqdm(range(S.n_labels + 1, 2 * S.n_labels + 1), desc="Computing expanded masks"):
                map_pf_im_2_bitmask_[i] = FU.compute_expanded_mask(map_pf_im_2_bitmask_[i - S.n_labels])

            np.save(path, map_pf_im_2_bitmask_)
            return

        with multiprocessing.Pool(settings.PARALLEL) as pool:
            with tqdm.tqdm(total=S.n_images, desc="Computing downsampled masks") as progress_bar:
                for (im_i, mask) in pool.imap_unordered(process_image, range(S.n_images)):
                    map_im_l_2_bitmask[im_i] = mask
                    progress_bar.update()

        print("Saving masks on disk (might take ~20 seconds)")

        if settings.EXPANSIONS:
            map_pf_im_2_bitmask_ = np.zeros((2 * S.n_labels + 1, S.n_images, *S.mask_shape), dtype=bool)
            map_pf_im_2_bitmask_[0: S.n_labels + 1] = np.swapaxes(map_im_l_2_bitmask, 0, 1)

            for i in tqdm.tqdm(range(S.n_labels + 1, 2 * S.n_labels + 1), desc="Computing expanded masks"):
                map_pf_im_2_bitmask_[i] = FU.compute_expanded_mask(map_pf_im_2_bitmask_[i - S.n_labels])
        else:
            map_pf_im_2_bitmask_ = np.swapaxes(map_im_l_2_bitmask, 0, 1)

        np.save(path, map_pf_im_2_bitmask_)

    def store_masks():
        leaves = {i : F.Leaf(i) for i in range(1, S.n_labels + 1)}

        R = range(1, 2 * S.n_labels + 1) if settings.EXPANSIONS else range(1, S.n_labels + 1)

        for pf_i in tqdm.tqdm(R, desc="Storing Label Masks"):
            new_p_formula = leaves[pf_i] if pf_i <= S.n_labels else F.Expand(leaves[pf_i - S.n_labels])
            all_p_formulas.append(new_p_formula)
            ScoreCalculator.known_masks[new_p_formula.to_code()] = map_pf_im_2_bitmask[pf_i]

    # -------------------------------------------------------------------------------------------------------------------------------
    # - "Densify" Neuron masks. I.e., if a mask has coverages like (0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 15, 0, 0, ...), -
    # -   turn them into (4, 6, 15, ...). This is *not* done for the creation of the annotation game files!                         -
    # -------------------------------------------------------------------------------------------------------------------------------
    if settings.EASY_MODE and densify:
        image_quantities = np.zeros(S.n_neurons, dtype=int)
        image_bitmasks = np.zeros((S.n_neurons, S.n_images), dtype=bool)

        for n_i in tqdm.tqdm(settings.NEURONS, desc="Computing truncated Neuron Masks (for Easy Mode)"):
            neuron_bitmasks = map_n_im_2_activations[n_i] > static_thresholds[n_i]
            n_of_images = 0

            for im_i in range(S.n_images):
                if neuron_bitmasks[im_i].sum() > 0:
                    map_n_im_2_activations[n_i, n_of_images] = map_n_im_2_activations[n_i, im_i]
                    n_of_images += 1
                    image_bitmasks[n_i, im_i] = True

            image_quantities[n_i] = n_of_images


    # ------------------------------------------------------------------------------------------------------------------------------------
    # - The file for expanded masks contains the regular masks. Thus, if the expanded file exists, this is sufficient even if            -
    # -   settings.EXPANSIONS is false. However, creating expanded masks is time consuming, so we don't always create the expanded file. -
    # -   This means we have to do some simple logic to figure out the correct approach.                                                 -
    # ------------------------------------------------------------------------------------------------------------------------------------
    downsampling = f"MD{settings.DOWNSAMPLING_THRESHOLD}" if settings.MANUAL_DOWNSAMPLING else "imresize"

    mask_file_path_expanded = f"results/masks_{downsampling}_X{settings.EXPANSION_THRESHOLD}.npy"
    mask_file_path_regular = f"results/masks_{downsampling}_regular.npy"
    path = mask_file_path_expanded if os.path.exists(mask_file_path_expanded) else mask_file_path_regular

    if not os.path.exists(mask_file_path_expanded) and (settings.EXPANSIONS or not os.path.exists(mask_file_path_regular)):
        create_masks()

    with open(path, "rb") as f:
        if path == mask_file_path_expanded and not settings.EXPANSIONS:
            map_pf_im_2_bitmask = np.load(f)[0: S.n_labels + 1]
        else:
            map_pf_im_2_bitmask = np.load(f)

    store_masks()

def process_image(im_i):

    def downsample(label_mask): # shape (S.labels + 1, 112, 112)
        if settings.MANUAL_DOWNSAMPLING:
            res = np.zeros(S.mask_shape, dtype=bool)

            for y in range(S.mask_shape[0]):
                for x in range(S.mask_shape[1]):
                    C = 0

                    for y_ in range(16):
                        for x_ in range(16):
                            C += label_mask[16 * y + y_, 16 * x + x_]

                    if C >= settings.DOWNSAMPLING_THRESHOLD:
                        res[y, x] = True
        else:
            return np.array(PIL.Image.fromarray(label_mask).resize(S.mask_shape, resample=PIL.Image.BILINEAR))

    def load_segmentation_image(path):
        rgb = imageio.imread(path)
        return rgb[:, :, 0] + rgb[:, :, 1] * 256

    full_mask = np.zeros((S.n_labels + 1, *S.mask_shape))

    for category_i, category in enumerate(MetaData.categories):
        paths_or_integers = MetaData.metadata_raw[im_i][category]

        if not paths_or_integers:
            continue  # Many images don't have all types of annotations

        if isinstance(paths_or_integers[0], int):  # segmentation = int for scene or texture
            integers = paths_or_integers

            for label in integers:
                if label == 0:
                    # Somehow 0 is a feature? Just continue - it exists in index.csv under scenes but not in c_scenes
                    continue
                full_mask[label] = True
            continue

        # At this point, we have a segmentation image or a list of segmentation images
        paths = paths_or_integers
        segmentation_images = np.array([load_segmentation_image(os.path.join(settings.IMAGE_DIRECTORY, path)) for path in paths])

        # For unity, make it so it's always a list
        if len(np.shape(segmentation_images)) == 2:
            segmentation_images = np.array([segmentation_images])

        labels = np.unique(segmentation_images.ravel())  # deletes duplicates and flattens array
        for label in labels:
            if label > 0:
                for segmentation_image in segmentation_images:
                    full_mask[label] = np.logical_or(full_mask[label], downsample(segmentation_image == label))

    return im_i, full_mask

def store_easy_masks(n_i):
    print(f"Storing Easy Masks for neuron {n_i}")
    for primitive_formula in all_p_formulas:
        mask = ScoreCalculator.known_masks[primitive_formula.to_code()][:]
        count = 0

        for j, coverage_bool in enumerate(image_bitmasks[n_i]):
            if coverage_bool:
                mask[count] = mask[j]
                count += 1
        assert image_quantities[n_i] == count

        ScoreCalculator.known_masks[primitive_formula.to_code(neuron_i=n_i)] = mask[0:image_quantities[n_i]]

def delete_easy_masks(n_i):
    for primitive_formula in all_p_formulas:
        del ScoreCalculator.known_masks[primitive_formula.to_code(neuron_i=n_i)]