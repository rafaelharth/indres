"""
STANDALONE CLASS.
Creates the annotation game files and folders.
"""

import tqdm
import numpy as np

import loaders.model_loader as ModelLoader
import loaders.neuron_loader as NeuronLoader
import loaders.mask_loader as MaskLoader
import score_calculator as ScoreCalculator
import loaders.metadata_loader as MetaData
import visualization.annotation_game as AnnotationGame
import settings
import util.util
import formulas.formula as F
import formulas.utils as FU
import util.util as Util
import util.information_printer as PI


FILTERS = [353, 69, 329, 154]
ACCESS = [True, False, False, True]

# The annotation game is why the easy mode exists! Doesn't make sense to run it without easy mode.
assert settings.EASY_MODE

def is_duplicate(index, array_of_arrays):
    count = 0

    for array in array_of_arrays:
        if index in array:
            count += 1

            if count > 1:
                return True

    return False

# ------------------------------------------------------------------------------------------------------------
# - Begin by creating all necessary directories that will be needed to avoid any potential for future errors -
# ------------------------------------------------------------------------------------------------------------
settings.init()

Util.create_directories(['results', 'results/annotation_game'])


# ---------------------------------------------------------------------------------------------------------
# - Call the init() function of all important classes to do whatever needs to be done to get them set up. -
# ---------------------------------------------------------------------------------------------------------
F.init()
FU.init()
MetaData.init()
NeuronLoader.init()


# ------------------------------------------------------------------------------------------------------------------------------------
# - Get the neuron activations (map_im_n_2_activations) that we wish to predict. To do this, use the register_forward_hook function. -
# ------------------------------------------------------------------------------------------------------------------------------------
model = ModelLoader.load_model(NeuronLoader.hook_function, hook_modules=[])
map_n_im_2_activations = NeuronLoader.get_neuron_values(model)


# --------------------------------
# - Calculate static thresholds. -
# --------------------------------
static_thresholds = util.util.compute_static_thresholds(map_n_im_2_activations)




# ------------------------------
# - Calculate the predictions. -
# ------------------------------
results = ScoreCalculator.run(map_n_im_2_activations, static_thresholds)


# ---------------------------------------------------------------------------------------------------------------------
# - Compute disjoint image sets. (This is horribly inefficient but still quick so there is no reason to optimize it.) -
# ---------------------------------------------------------------------------------------------------------------------
filter_indices = [[] for _ in FILTERS]

for i, filter_i in enumerate(filter_indices):
    print(static_thresholds[FILTERS[i]])
    bitmask = map_n_im_2_activations[FILTERS[i]] > static_thresholds[FILTERS[i]]
    filter_indices[i] = np.where(bitmask.sum((1,2)) > 0)[0]
    PI.show_list(list(filter_indices[i]))
    if 54 in filter_indices[i]: # Annoying but I've used this as an example in the outgoing emails so it can't be a test image
        filter_indices[i] = np.setdiff1d(filter_indices[i], np.array([54]))



tainted = np.array([True for _ in filter_indices])

while np.any(tainted):
    # Find largest tainted list
    max_n_indices = 0
    max_index = -1

    for i, indices in enumerate(filter_indices):
        if tainted[i] and len(indices) > max_n_indices:
            max_index = i
            max_n_indices = len(indices)

    # largest tainted list now has index max_index. Do one step on purifying it.

    found_tainted = False
    for image_index in filter_indices[max_index]:
        if is_duplicate(image_index, filter_indices):
            filter_indices[max_index] = np.setdiff1d(filter_indices[max_index], np.array([image_index]))
            found_tainted = True
            break

    if not found_tainted:
        tainted[max_index] = False

print("After removing duplicates...")
for i, filter_ in enumerate(FILTERS):
    print(f"Filter {filter_} has {len(filter_indices[i])} images")
    PI.show_list(list(filter_indices[i]))
    np.random.shuffle(filter_indices[i])



# -----------------------------------------------------------------------------------------------------
# - DO NOT COMPRESS THE REPRESENTATION VIA (0,0,4,0,0,0,6,0,...) -> (4,6,...). (Hence densify=False.) -
# - ALL OF THE ANNOTATION GAME JAVASCRIPT FILES DEAL WITH REGULAR UNCHANGED ABSOLUTE INDICES.         -
# -----------------------------------------------------------------------------------------------------
MaskLoader.init(map_n_im_2_activations, static_thresholds, densify=False)


# ------------------------------------------------
# - Create files for the manual annotation game. -
# ------------------------------------------------
for filter_i, filter_ in tqdm.tqdm(enumerate(FILTERS), desc="Creating Annotation Game files"):
    T = static_thresholds[filter_]
    AnnotationGame.create_html(filter_, map_n_im_2_activations[filter_], T, results, 'test', filter_indices[filter_i], ACCESS[filter_i])
    AnnotationGame.create_html(filter_, map_n_im_2_activations[filter_], T, results, 'train', filter_indices[filter_i], ACCESS[filter_i])

