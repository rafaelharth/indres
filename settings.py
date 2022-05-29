"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

The interface to change program settings
"""

import score_calculator as ScoreCalculator
import formulas.formula as F


# ----------------------------
# - Most important settings: -
# ----------------------------

# Current settings are for replicating results from the paper at formula length 3.
NEURONS = list(range(512))
# OPERATOR_SET = [F.Or, F.And, F.AndNot]
OPERATOR_SET = [F.Or, F.And, F.AndNot, F.With, F.Close]
SCORE_FUNCTION_OPTIMIZATION = "imroun" # either iou or imroun is recommended
SCORE_FUNCTION_REPORT = "imroun"
N_BEAMSEARCH_CANDIDATES = 50 # 40-50 is recommended if you care about the quality of results
BEAM_SIZE = 4 # 4 or more is recommended
MAX_FORMULA_LENGTH = 3
MANUAL_DOWNSAMPLING = True # I think Manual Downsampling creates higher quality masks but it's more expensive
EXPANSIONS = True
JUMPSTART = False
DYNAMIC_THRESHOLDS = False
PRINT_INFORMATION = True
EASY_MODE = False


# ---------------------------------
# - Next most important settings: -
# ---------------------------------
MULTIPLIER_IP = 0.75 # the r in imrou_r. Empirically, .75 seems very reasonable. 1 is definitely too high. 0.5 does seem too low.
EXPANSION_THRESHOLD = 3 # regulates EXPAND operator. 2 had higher coverages which I don't like. 4+ it will probably make the score worse; 1 is out of the question.
DOWNSAMPLING_THRESHOLD = 100 # While 16*16/2 = 118, to me 100 feels intuitively better when looking at masks and it leads to higher scores
IGNORE_RELATIVE_COVERAGE_THRESHOLD = 0.0001 # all formulas with coverage less than this are ignored outright. Should be correspondingly low
ABORT_CLOSE_FORMULAS = True # Settings for early abortion. This decreases runtime by about half.
CUTOFF_SAMPLES = 4000 # There is probably a way to compute this and the next value along with probability guarantees using statistical methods. However, this is
CUTOFF_SCORE = 0.5 #       very hard (what's the variance of our RVs?), so I looked at some sample results instead, yielding 4000 / 0.5 as reasonable values
PARALLEL = 8  # Number of processes that are running beam search concurrently. Probably should equal the number of cores on the machine.
UPDATE_THRESHOLDS_IMMEDIATELY = True # - This & next 3 only relevant if DYNAMIC_THRESHOLDS is True -
STARTING_THRESHOLD = 2 # three now deprecated options for dynamic thresholds
THRESHOLD_STEP = 0.1
THRESHOLD_MIN_IMPROVEMENT = 0.0001


# ------------------------------------------------------------------------------------------------------------------------------------------
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - EXPLANATIONS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# ------------------------------------------------------------------------------------------------------------------------------------------
# - NEURONS: the set of neurons for which to compute results. Must be a subset of {0, ..., 512}.                                           -
# -                                                                                                                                        -
# - OPERATOR_SET: the set of operators to consider during beam search. The standard algorithm uses {OR, AND, AND NOT}, the close algorithm -
# -               uses that plus {WITH, CLOSE}. The operator CLOSE-TO must not be added as it will be considered iff CLOSE is present.     -
# -                                                                                                                                        -
# - SCORE_FUNCTION_OPTIMIZATION: the function used to compute the scores that decide which formulas are kept in the beam (and which        -
# -                              formula is ultimately returned). Note that the difference between imrou and imroun has no effect here.    -
# -                                                                                                                                        -
# - SCORE_FUNCTION_REPORT: the function used to compute the scores displayed in the results (as "Score")                                   -
# -                                                                                                                                        -
# - N_BEAMSEARCH_CANDIDATES: at the beginning of the beam search, the candidate space for all neurons is artificially restricted to only   -
# -                          a subset of all labels. This variable determines the size of that subset. If set to None, there will be no    -
# -                          restriction. However, note that this would multiply runtime by about 400 or 800 (depending on EXPANSIONS).    -
# -                          It's rare for a formula not to be found if the value is around 50.                                            -
# -                                                                                                                                        -
# - IGNORE_RELATIVE_COVERAGE_THRESHOLD: formulas with coverage smaller than [this variable] * [total image space] are ignored during beam  -
# -                                     search. This will be forced to 0 if EASY_MODE is set to True.                                      -
# -                                                                                                                                        -
# - BEAM_SIZE: the number of formulas we keep in the beam after each pass.                                                                 -
# -                                                                                                                                        -
# - MAX_FORMULA_LENGTH: the maximum formula length that can be found during beam search. After determining the candidate space, the beam   -
# -                     will run MAX_FORMULA_LENGTH - 1 passes through the candidate space. It's rare for an output formula to be shorter  -
# -                     than the maximum length.                                                                                           -
# -                                                                                                                                        -
# - MANUAL_DOWNSAMPLING: determines how annotations are downsampled to the 7-by-7 resolution of neuron activations. If set to False, the   -
# -                      library function PIL.Image.resize([mask_name], resample=PIL.Image.BILINEAR) is used. If set to True, block-based  -
# -                      downsampling is used, i.e, each 112-by-112 mask is partitioned into seven 16-by-16 blocks, and for each block,    -
# -                      the respective cell in the 7-by-7 grid is set to True iff at least N cells of the 16-by-16 block are True. The    -
# -                      value of N is regulated by DOWNSAMPLING_THRESHOLD. While 128 is the most natural (setting the threshold at 50% of -
# -                      all cells in a 16-by-16 block, I've found from inspecting masks that I prefer using a value of 100. Note that     -
# -                      manual downsampling is much more computationally expensive than bilinear downsampling.                            -
# -                                                                                                                                        -
# - EXPANSIONS: if set to True, the initial set of 1198 label masks is extended by their 1198 expanded versions (even though this has no   -
# -             effect scene-level masks). The EXPAND operator is not used afterward during beam search, but this is only because it is    -
# -             too computationally expensive. Given unlimited computing power, it would be treated like any other operator.               -
# -                                                                                                                                        -
# - JUMPSTART: if set to True, progress from previous runs of the program stored in jumpstart/jumpstart.txt is utilized (see               -
# -            jumpstart/__init__.py for an extensive explanation). This can be used to distribute computation across several settings     -
# -            with only minor overhead.                                                                                                   -
# -                                                                                                                                        -
# - DYNAMIC_THRESHOLDS: (DEPRECATED) determines how neuron values are thresholded to compute neuron activations. If set to False, static   -
# -                     thresholds are used such that each neuron has total coverage of 0.5%. If set to True, the threshold is optimized   -
# -                     throughout the beamsearch to maximize the score (this happens for each neuron independently). If this is combined  -
# -                     with regular iou as the metric, thresholds will tend to be chosen extremely low so as to maximize coverage, but it -
# -                     has produced reasonable results if combined with imrou scores. However, I haven't used this in a long time because -
# -                     runtimes become infeasible if it is combined with CLOSE operators, and it probably doesn't work anymore.           -
# -                                                                                                                                        -
# - PRINT_INFORMATION: if set to True, interesting stuff printed onto the console throughout the beam search. This is required to make use -
# -                    of the jumpstart functionality.                                                                                     -
# -                                                                                                                                        -
# - EASY_MODE: if set to True, for each neuron, the set of images on which it is scored is reduced to only contain those where the neuron  -
# -            activates on at least one cell. This is primarily relevant to compare algorithmic scores with scores from human annotators. -
# -            (Without this setting, about 95% of images have no activations for every neuron, making it vastly less fun for humans to    -
# -            play the annotation game.)                                                                                                  -
# ------------------------------------------------------------------------------------------------------------------------------------------
# -                                                                                                                                        -
# - ABORT_CLOSE_FORMULAS: if set to True, the algorithm will begin mask computation of Close formulas by only considering a subset of all  -
# -                       images. If the score on those is sufficiently bad, computation is terminated early. The parameters regulating    -
# -                       the size of this subset and the definition of 'sufficiently bad' are CUTOFF_SAMPLES and CUTOFF_SCORE. The        -
# -                       current settings reduce runtime by about half with very small (I think ~0.1%) chances for false abortions.       -
# ------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------
# - I  = intersection        -- F -- IoU   -
# - U  = union               -- N -- ImR   -
# - R  = random intersection -- C --       -
# - o  = over                -- N -- ImRoU -
# - m  = minus               -- S -- IoR   -
# ------------------------------------------

COMPLEXITY_PENALTY = 0.001 # miniscule penalty for long formulas

# Beam search params
EXTRA_LOW_PENALTY_LABELS = 0 # Number of additional labels (technically primitive formulas), computed without penalty, for the initial beam

QUANTILE = 0.005  # Compute static thresholds according to this quantile

GPU = False  # ImE, this doesn't make a lot of difference because only a very small part of the runtime goes into running the model
INDEX_FILE = "index_ade20k.csv"  # Which index file to use? If _sm, use test mode
CLEAN = False  # set to "True" if you want to clean the temporary large files after generating result
MODEL = "resnet18"  # only resnet is supported

PROBE_DATASET = "broden"  # currently only broden dataset is supported
SCORE_THRESHOLD = 0.01  # the threshold such that neurons with scores below it are shown in gray
SAMPLE_N = 8  # number of images sampled for each neuron

OUTPUT_FOLDER = "" # This is set in init because circular imports problem


# ----------------------------------------------------------------------------------------------------------------------------------
# - I've removed all of the logic that sets these parameter value for different models or layers or whatnot.                       -
# - For the purposes of the paper I wanted to write, there is one dataset (places365) and one layer (penultimate layer of resnet). -
# ----------------------------------------------------------------------------------------------------------------------------------
DATA_DIRECTORY = "dataset/broden1_224"
IMAGE_DIRECTORY = "dataset/broden1_224/images"
IMG_SIZE = 224
NUM_CLASSES = 365
FEATURE_NAMES = ['layer4']
MODEL_FILE = f"zoo/{MODEL}_places365.pth.tar"
MODEL_PARALLEL = True

WORKERS = 12
BATCH_SIZE = 128
TALLY_BATCH_SIZE = 16
TALLY_AHEAD = 4



# -----------------------------------------------------------------------------------------------------------------------------------------
# - I still do not understand what determines whether circular imports cause a problem. (There are no from {} import {} statements left.) -
# - Whatever the case, setting SCORE_FUNCTION_OPTIMIZATION directly does cause one. This is a highly inelegant workaround.                -
# -----------------------------------------------------------------------------------------------------------------------------------------
def init():
    global SCORE_FUNCTION_OPTIMIZATION, SCORE_FUNCTION_REPORT, OUTPUT_FOLDER, ABORT_CLOSE_FORMULAS, IGNORE_RELATIVE_COVERAGE_THRESHOLD

    if SCORE_FUNCTION_OPTIMIZATION == "iou":
        SCORE_FUNCTION_OPTIMIZATION = ScoreCalculator.iou
        function_string = "iou"
    elif SCORE_FUNCTION_OPTIMIZATION == "imrou":
        SCORE_FUNCTION_OPTIMIZATION = ScoreCalculator.imrou
        function_string = f"imrou{int(MULTIPLIER_IP * 100)}"
    elif SCORE_FUNCTION_OPTIMIZATION == 'imroun':
        SCORE_FUNCTION_OPTIMIZATION = ScoreCalculator.imroun
        function_string = f"imrou{int(MULTIPLIER_IP * 100)}"
    else:
        raise ValueError(f"Value {SCORE_FUNCTION_OPTIMIZATION} of optimization function not recognized.")

    if SCORE_FUNCTION_REPORT == "iou":
        SCORE_FUNCTION_REPORT = ScoreCalculator.iou
    elif SCORE_FUNCTION_REPORT == "imrou":
        SCORE_FUNCTION_REPORT = ScoreCalculator.imrou
    elif SCORE_FUNCTION_REPORT == 'imroun':
        SCORE_FUNCTION_REPORT = ScoreCalculator.imroun
    else:
        raise ValueError(f"Value {SCORE_FUNCTION_REPORT} of report function not recognized.")

    OUTPUT_FOLDER = f"results/FL{MAX_FORMULA_LENGTH}_{function_string}_{'X' + str(EXPANSION_THRESHOLD) if EXPANSIONS else 'N'}" \
                    f"_{'close' if F.Close in OPERATOR_SET else 'standard'}{'_EASY' if EASY_MODE else ''}"

    # ----------------------------------------------------
    # - Code for fixing inconsistent settings goes here. -
    # ----------------------------------------------------
    if EASY_MODE:
        ABORT_CLOSE_FORMULAS = False
        IGNORE_RELATIVE_COVERAGE_THRESHOLD = 0
