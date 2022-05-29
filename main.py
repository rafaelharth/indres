"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

The main method -- execute this to run the program. First creates necessary directories, then calls other functions in the proper order
"""

import os

import S
import score_calculator as ScoreCalculator
import formulas.formula as F
import formulas.utils as FU
import loaders.model_loader as ModelLoader
import loaders.neuron_loader as NeuronLoader
import util.util
import visualization.html_summary as HtmlSummary
import loaders.metadata_loader as MetaData
import loaders.mask_loader as MaskLoader
import settings
import util.util as Util


# -------------------------------------------------------------------------------------------------------------
# - Begin by creating all necessary directories that will be needed to avoid any potential for future errors. -
# -------------------------------------------------------------------------------------------------------------
settings.init()
L, R = min(settings.NEURONS), max(settings.NEURONS)
Util.create_directories(['results', settings.OUTPUT_FOLDER, os.path.join(settings.OUTPUT_FOLDER, f"images_{L}_{R}"), "jumpstart", "jumpstart/neurons"])



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


# ----------------------------------------------------------------------------------------------------------------------------------
# - If settings.EASY_MODE is True, the init function of the mask loader will require these two parameters, so we only call it now. -
# ----------------------------------------------------------------------------------------------------------------------------------
MaskLoader.init(map_n_im_2_activations, static_thresholds)



# ------------------------------
# - Calculate the predictions. -
# ------------------------------
results = ScoreCalculator.run(map_n_im_2_activations, static_thresholds)


# ------------------------------
# - Visualize the predictions. -
# ------------------------------
HtmlSummary.create(map_n_im_2_activations, results)


# --------------------------------------------------------
# - If the settings say so, remove the large .mmap file. -
# --------------------------------------------------------
if settings.CLEAN:
    file_list = [f for f in os.listdir(settings.OUTPUT_FOLDER) if f.endswith("mmap")]
    for f in file_list:
        os.remove(os.path.join(settings.OUTPUT_FOLDER, f))
