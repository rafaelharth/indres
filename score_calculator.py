"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

Implements the main functionality of this project: takes neuron activations & thresholds and computes the best approximate formulas
based on label masks. The formulas are saved under "results/FL{formula_length}_{optimization_metric}_{identifier}_{algorithm}/",
where identifier is either "X{n}" if EXPAND(n) labels are used (see settings.py) or N if not, and algorithm is either "standard" or "close"
"""

import copy
import os
import pickle
import random
import time

import pandas as pd
import multiprocessing as mp
import numpy as np
import random as rand
from tqdm import tqdm
from collections import Counter

import S
import settings
import util.information_printer as PI
import formulas.formula as F
import formulas.utils as FU
import formulas.parser as Parser
import loaders.metadata_loader as MetaData
import loaders.mask_loader as MaskLoader

c_iou = 0
c_formulas = 0

# ---------------------------------------------------------------------------------------------------------------------------------------
# - Globals for multiprocessing to prevent shared memory. Note: this only works properly under Linux due to how forking is implemented. -
# ---------------------------------------------------------------------------------------------------------------------------------------
g = {}

# ----------------------------------------------------------------------------------------------------------------------
# - Maps the codes of known formulas to their masks. (It has to be codes so it's applicable across different neurons.) -
# ----------------------------------------------------------------------------------------------------------------------
manager = mp.Manager()
known_masks = manager.dict()


def run(map_n_im_2_activations, static_thresholds):
    """
    Params:
        map_n_im_2_activations: neuron activations across the entire dataset (shape: n_neurons n_images x *activation_map)

    Computes binary masks for neurons, measures scores (IoU or variations) w/ compositional search,
      and saves resulting formulas under results.
    """

    def process_formula_records():
        # ---------------------------------------------------------------------------------
        # - Parse the formula_records.pkl file and extract relevant information.          -
        # - Currently, this includes only keeping track of which formulas have zero area. -
        # ---------------------------------------------------------------------------------
        path_empty_formulas = 'record/empty_formulas.pkl'

        with open(path_formula_records, 'rb') as f:
            formula_record_dict = pickle.load(f)

        empty_formulas = formula_record_dict["empty"] if "empty" in formula_record_dict else set()
        g["empty_formulas"] = empty_formulas

        if os.path.exists(path_empty_formulas):
            with open(path_empty_formulas, 'rb') as f:
                M_ = pickle.load(f)
            empty_formulas.union(M_)


        with open(path_empty_formulas, 'wb') as f:
            pickle.dump(empty_formulas, f)

    path_results = os.path.join(settings.OUTPUT_FOLDER, f"results_{min(settings.NEURONS)}_{max(settings.NEURONS)}.csv")
    path_history = os.path.join(settings.OUTPUT_FOLDER, f"history_{min(settings.NEURONS)}_{max(settings.NEURONS)}.txt")

    g["inspected_neurons"] = map_n_im_2_activations
    g["static_thresholds"] = static_thresholds

    # If the results already exist, we read them from the file instead of proceeding with the method
    if path_results and os.path.exists(path_results):
        print(f"Returning cached results from {path_results}.")
        return MetaData.load_csv(path_results)


    g["labels"] = [label_i + 1 for label_i in range(S.n_labels)]
    g["pos_p_formulas"] = [p_formula for p_formula in MaskLoader.all_p_formulas if known_masks[p_formula.to_code()].sum() > 0]

    if settings.PRINT_INFORMATION:
        dashes = PI.show("Printing some label names")

        for label in g["labels"]:
            if rand.random() < 0.01:
                print(f"label {label} is called {MetaData.name(label)}")
        PI.aftermath(dashes)


    path_formula_records = 'record/formula_records.pkl'
    if os.path.exists(path_formula_records):
        process_formula_records()


    category_namer = lambda label: 'NONE' if label is None else MetaData.categories[MetaData.map_label_2_max_coverage_c_i[label]]
    records = []
    histories = {}
    map_neuron_i_2_formula_records = {}

    with mp.Pool(settings.PARALLEL) as pool:
        with tqdm(total=len(settings.NEURONS), desc="Formula Beam Search") as progress_bar:
            for (neuron_i, formula, threshold, formula_records, history) in pool.imap_unordered(beamsearch, settings.NEURONS):
                print(f"{c_formulas:.1f} / {c_iou:.1f}")
                histories[neuron_i] = history
                w_category = formula.to_str(category_namer)
                map_neuron_i_2_formula_records[neuron_i] = formula_records

                records.append({"neuron": neuron_i, "category": w_category, "formula": formula.to_str(), "threshold": f'{threshold:0.1f}'})

                progress_bar.update()

                if not os.path.exists(f'jumpstart/neurons/{neuron_i}.txt'):
                    with open(f'jumpstart/neurons/{neuron_i}.txt', 'w') as f:
                        f.write(f"\n\n ||||||| {neuron_i} |||||||")
                        f.write("\n".join(history))

    pool.join()
    records = sorted(records, key=(lambda R: R["neuron"]))
    tally_df = pd.DataFrame(records)
    tally_df.to_csv(path_results, index=False)

    if settings.PRINT_INFORMATION and F.Close in settings.OPERATOR_SET:
        PI.show(f"Total time spent in CloseTo methods: ~{sum(FU.TIMES):.0f}) seconds (across all {settings.PARALLEL} processes).")
        if settings.ABORT_CLOSE_FORMULAS:
            PI.show(f"Finished about {10*sum([x for x in FU.ABORTIONS if x == 1])} out of {10*len(FU.ABORTIONS)} CLOSE-TO computations.")

    PI.show_simple(f"Attempting to dump the formula records file under {path_formula_records}.")

    with open(path_formula_records, "wb") as f:
        pickle.dump(map_neuron_i_2_formula_records, f)

    with open(path_history, 'w') as f:
        for (neuron_i, history) in histories.items():
            f.write(f"\n\n ||||||| {neuron_i} |||||||")
            f.write("\n".join(history))

    return records

def beamsearch(neuron_i):
    """
    Compute best label and score for the given neuron via beam search.
    """
    # ---------------------------------------------------------------------------------------------------------------
    # - The following internal functions make heavy use out of the shared namespace and cannot be read in isolation -
    # ---------------------------------------------------------------------------------------------------------------
    def print_beam():
        dashes = PI.show(f"For neuron {neuron_i} at formula length {formula_length}, the beam is", history=history)
        PI.show_list([f"{f.to_str()}, s={score:.4f}/th={th:.1f}" for (f, (score, th)) in beam.items()], max_length=120, history=history)
        PI.aftermath(dashes, history=history)

    def compute_masks_for_beam_formulas():
        for formula in beam.keys():
            code = formula.to_code(neuron_i=neuron_i)
            if code not in known_masks.keys():
                known_masks[code] = FU.compute_composite_mask(formula, neuron_i=neuron_i_when_computing_l_mask)
        PI.show_simple(f"  (currently memorizing {len(known_masks)} masks.)")

    def initialize_beam(formulas):
        beam_ = {}
        codes_ = set()
        while len(beam_) < settings.BEAM_SIZE:
            formulas = {formula: R for (formula, R) in formulas.items() if formula.to_code(ignore_expansion=True) not in codes_}
            I = np.argmax([score for (score, _) in formulas.values()])
            formula, (score, threshold) = list(formulas.items())[I]
            beam_[formula] = (score, threshold)
            codes_.add(formula.to_code(ignore_expansion=True))
            if not isinstance(formula, F.Leaf) and not isinstance(formula, F.Expand):
                stored_mask_codes.add(formula.to_code(neuron_i=neuron_i))

        return beam_

    def compute_redundancy_bitmask():
        RS = []

        for new_formula in new_formulas:
            # check for (1) repetition, (2) syntactic redundancy of the form (A op A) (3) s/r of the form (A or B) or A
            redundancy, reason = FU.redundancy_check(formula, new_p_formula, new_formula, op, codes)
            if redundancy:
                if random.random() < 0.01:
                    PI.show_simple(f"N#{neuron_i}: formula {new_formula.to_str()} redundant due to {reason} (1 of 100)", history=history)
                RS.append(True)
            elif "empty_formulas" in g and new_formula.to_code() in g["empty_formulas"]:  # check for empty area
                if random.random() < 0.001:
                    PI.show_simple(f"Formula {new_formula.to_str()} has tiny area (1 of 1000)", history=history)
                RS.append(True)
            else:
                RS.append(False)

        # ----------------------------------------------------------------------------------------------------------------------
        # - Additionally, check for redundancies due to scene-level annotations. If B is a primitive formula of a scene label, -
        # -   then (A CLOSE TO B) == (A AND B), whereas (A CLOSE B) == (B CLOSE TO A), so we don't need to include either one. -
        # ----------------------------------------------------------------------------------------------------------------------
        label = new_p_formula.val if isinstance(new_p_formula, F.Leaf) else new_p_formula.val.val
        category = MetaData.categories[MetaData.map_label_2_max_coverage_c_i[label]]

        if op == F.Close and len(RS) == 3:
            if category == 'scene':
                RS[0] = True  # A close SCENE
                RS[1] = True  # A close to SCENE
            elif isinstance(formula, F.Leaf) and category == 'scene':
                RS[0] = True  # SCENE close A
                RS[2] = True  # A close to SCENE
        elif op in [F.Close, F.With] and len(RS) == 1:
            if category == 'scene':
                RS = [True]  # A close SCENE or A with SCENE

        return RS

    def compute_formulas_w_positive_score():
        global c_formulas, c_iou
        formulas = {}

        for p_formula in g["pos_p_formulas"]:
            # print(p_formula)
            if isinstance(p_formula, F.Expand) and MetaData.categories[MetaData.map_label_2_max_coverage_c_i[p_formula.val.val]] == 'scene':
                continue # We do not need expanded scenes

            x = time.time()
            label_mask = FU.compute_composite_mask(p_formula, neuron_i=neuron_i_when_computing_l_mask)
            c_formulas += time.time() - x


            x = time.time()
            score = settings.SCORE_FUNCTION_OPTIMIZATION(static_neuron_mask, label_mask)
            c_iou += time.time() - x

            formulas[p_formula] = (score, g["static_thresholds"][neuron_i])

        formulas = {F_: (score, th) for F_, (score, th) in formulas.items() if score > 0}

        return formulas # DO NOT INLINE. The point here is to delete the large dictionary to save memory

    def try_parsing_jumpstart_file():
        """
        returns None for all elements that it can't obtain
        """

        if not settings.JUMPSTART or not os.path.exists(f"jumpstart/jumpstart.txt"):
            return None, None, None

        with open(f"jumpstart/jumpstart.txt", 'r') as f:
            lines = f.readlines()
            R_ = {}
            candidates = None

            # --------------------------------------------------------------------------------------------------------------
            # - We're parsing history dumps that look like this for candidate sets (with lines truncated at |):            -
            # -                                                                                                            -
            # -         Neuron #23 has candidate set:                                                                      -
            # -         -----------------------------                                                                      -
            # -         kitchen-s, cabinet-X, cabinet, stove-X, youth hostel-s, work surface-X, refrigerator-X, stove, do| -
            # -         door, microwave-X, oven-X, oven, microwave, kitchen island-X, cubicle-office-s, refrigerator, des| -
            # -         kitchen island, bed-X, wall-X, desk, dorm room-s, autobus-X, drawer-X, bed, booth, wall, grey-c-X| -
            # -         brown-c-X, orange-c-X, dinette-vehicle-s, parking garage-indoor-s, wardrobe-X, sink-X, ladder-X, | -
            # -         bus interior-s, brown-c, orange-c, fire escape-s, telephone booth-X, black-c-X, dishwasher-X, win| -
            # -         dishwasher, wet bar-s, counter-X, button panel-X,                                                  -
            # -         ----------------------------                                                                       -
            # -                                                                                                            -
            # - and this for formula lengths:                                                                              -
            # -                                                                                                            -
            # -         For neuron 3 at formula length 1, the beam is                                                      -
            # -         ---------------------------------------------                                                      -
            # -         auditorium-s, s=0.0840 / th=4.6, seat-X, s=0.0786/th=4.6, movie theater-indoor-s, s=0.0709/th=4.6, -
            # -         silver screen-X, s = 0.0556/th=4.6,                                                                -
            # -         ---------------------------------------------                                                      -
            # -                                                                                                            -
            # --------------------------------------------------------------------------------------------------------------

            for i, line in enumerate(lines):
                if "has candidate set" in line:
                    contents = line.split(' ')
                    if int(contents[1][1:]) == neuron_i:
                        candidate_text = ""
                        lookahead = i+2

                        while "-----" not in lines[lookahead]:  # stop once we hit the ----- line after the beam text
                            candidate_text += lines[lookahead]
                            lookahead += 1

                        formula_strings = candidate_text.split(',')
                        candidates = [Parser.parse(FS.strip()) for FS in formula_strings if FS.strip() != ""]

                if "at formula length" in line:
                    contents = line.split(' ')
                    if int(contents[2]) == neuron_i:
                        beam_text = ""
                        formula_length = int(contents[6][:-1])
                        lookahead = i+2 # skip to first line of beam text

                        while "-----" not in lines[lookahead]: # stop once we hit the ----- line after the beam text
                            beam_text += lines[lookahead]
                            lookahead += 1

                        R_[formula_length] = beam_text

            R_ = {FL: value for (FL, value) in R_.items() if FL <= settings.MAX_FORMULA_LENGTH}

            if len(R_) == 0:
                # Didn't find a beam, so just return the candidates (or None of we didn't fine those, either)
                return candidates, None, None
            else:
                # The history probably contains a bunch of different beams. Pick the right one.
                relevant_formula_length = max([FL for FL in R_.keys() if FL <= settings.MAX_FORMULA_LENGTH])
                contents = R_[relevant_formula_length].split(',')

                # At this point, contents should be something like ["auditorium-s", " s=0.0840/th=4.6", " seat-X", ...]. Process it now.
                formula_strings = [FS.strip() for (i, FS) in enumerate(contents) if i % 2 == 0 and FS.strip() != ""]
                formula_data = [FD.strip() for (i, FD) in enumerate(contents) if i % 2 == 1 and FD.strip() != ""]

                def parse_data_string(data_string):
                    # ------------------------------------------------------------
                    # - The kind of string we are parsing has the form           -
                    # -            s=0.116/th=5.1                                -
                    # - What follows is some tedious character-level arithmetic. -
                    # ------------------------------------------------------------
                    slash_index = data_string.index('/')

                    return float(data_string[2:slash_index]), float(data_string[slash_index + 4:])

                beam_ = {Parser.parse(FS) : parse_data_string(DS) for (FS, DS) in zip(formula_strings, formula_data)}

                return candidates, beam_, relevant_formula_length

    # -------------------
    # - Start of method -
    # -------------------
    formula_records = []
    history = []
    stored_mask_codes = set()
    threshold_ = g["static_thresholds"][neuron_i]
    static_neuron_mask = g["inspected_neurons"][neuron_i] > threshold_
    neuron_i_when_computing_l_mask = neuron_i if settings.EASY_MODE else None

    if settings.EASY_MODE:
        MaskLoader.store_easy_masks(neuron_i)

    # ------------------------------------------------------------------------------------------------------------------------------------
    # - Check if there are records available for the candidate set and the beam. If so, read both from out of the jumpstart file instead -
    # -   of computing them the hard way. After this call, all variables we couldn't obtain from the file will be None.                  -
    # ------------------------------------------------------------------------------------------------------------------------------------
    beam_search_candidates, beam, formula_length = try_parsing_jumpstart_file()

    # all the runtime up to the real beamsearch is from here--
    if beam is None or beam_search_candidates is None:
        formulas_w_pos_score = compute_formulas_w_positive_score()
        if not formulas_w_pos_score:
            raise ValueError(f"Neuron {neuron_i} has no labels with positive score.")
    # -- to here

    if beam_search_candidates is None:
        beam_search_candidates = [formula for formula, _ in Counter(formulas_w_pos_score).most_common(settings.N_BEAMSEARCH_CANDIDATES)]

        # Print the candidate set
        dashes = PI.show(f"Neuron #{neuron_i} has candidate set:", history=history)
        PI.show_list([formula.to_str() for formula in beam_search_candidates], history=history)
        PI.aftermath(dashes, history=history)

    if beam is None:
        beam = initialize_beam(formulas_w_pos_score)
        formula_length = 1
        print_beam()


    # ------------------------
    # - Run the Beam search. -
    # ------------------------
    codes = [formula.to_code(neuron_i=neuron_i) for formula in beam.keys()]

    while formula_length < settings.MAX_FORMULA_LENGTH:
        formula_length += 1
        compute_masks_for_beam_formulas()


        # --------------------------------------------------------------------------------------------------------------------------------
        # - Aborting early means that, given a formula, rather than computing the entire mask (i.e., across all images) for the formula  -
        # -   right away, we first compute it for a subset of images. If the score is sufficiently low for that subset, we abort         -
        # -   computation there. In this case, we return None instead of the mask. The point at which we check and the score cutoff as a -
        # -   proportion of the worst score in the beam are determined by settings.CUTOFF_SAMPLES and settings.CUTOFF_SCORE.             -
        # --------------------------------------------------------------------------------------------------------------------------------
        if settings.ABORT_CLOSE_FORMULAS:
            worst_current_score = np.min([score for score, _ in beam.values()])
            PI.show_simple(f"Neuron #{neuron_i}: worst score={worst_current_score}", history=history)
            abort_data = (settings.CUTOFF_SAMPLES, settings.CUTOFF_SCORE * worst_current_score, static_neuron_mask)
        else:
            abort_data = None

        beam_with_new_formulas = copy.deepcopy(beam) # initialize as current beam

        for formula in beam.keys():
            for new_p_formula in beam_search_candidates:
                for op in settings.OPERATOR_SET: # Usually either [F.Or, F.And, F.AndNot, F.With, F.Close] or just the first three

                    # new_formulas will either be of the form [] or [x] or [x,y,z]                                     -
                    new_formulas = FU.list_new_formulas(formula, new_p_formula, op)


                    # -------------------------------------------------------------------------------------------------------------------
                    # - To save computational cost, check for redundancies, i.e., information that guarantees a formula won't be novel. -
                    # -------------------------------------------------------------------------------------------------------------------
                    redundancies = compute_redundancy_bitmask()

                    if all(redundancies):
                        continue

                    if op == F.Close and len(new_formulas) == 3: # Need A CLOSE B and A CLOSE TO B and B CLOSE TO A; compute them jointly
                        close_masks = FU.compute_close_masks(formula, new_p_formula, redundancies, abort=abort_data, neuron_i=neuron_i)
                    else:
                        close_masks = None


                    # -----------------------------
                    # - Process all new formulas. -
                    # -----------------------------
                    for i, new_formula in enumerate(new_formulas):
                        # This gives wrong results when I replaced with "close_formulas and not close_formulas[i]"; I don't know why.
                        if redundancies[i] or (close_masks is not None and close_masks[i] is None):
                            continue


                        # ------------------------------------------------------------------------------------------------------------------
                        # - Now compute the new label mask. Don't need to differentiate cases as compute_composite_mask will check whether -
                        # -    or not the required masks have been stored under known_masks.                                               -
                        # ------------------------------------------------------------------------------------------------------------------
                        new_label_mask = FU.compute_composite_mask(new_formula, abort=abort_data, neuron_i=neuron_i_when_computing_l_mask)


                        # --------------------------------------------------------------------------------------------------
                        # - Here is where we check whether abortion has taken place or not. If so, new_label_mask is None. -
                        # --------------------------------------------------------------------------------------------------
                        if new_label_mask is None:
                            continue

                        # ----------------------------------------------------------------------------------------------------------------
                        # - As another performance saving mechanism, check for small coverage. While this will catch fewer redundant     -
                        # -   formulas than measuring the score, it has the advantage of being independent of the neuron, which means we -
                        # -   can keep a global list of all such formulas and check for redundancy before mask computation even begins.  -
                        # ----------------------------------------------------------------------------------------------------------------
                        if new_label_mask.sum() < S.n_pixels * settings.IGNORE_RELATIVE_COVERAGE_THRESHOLD:
                            formula_records.append((new_formula.to_str(), None))
                            continue


                        # -----------------------------------------------------------------------------------------------------------------
                        # - Use the label mask to compute the score. Choose one of three implementations based on the threshold settings. -
                        # -----------------------------------------------------------------------------------------------------------------
                        if settings.DYNAMIC_THRESHOLDS:
                            _, threshold = beam[formula]

                            if settings.UPDATE_THRESHOLDS_IMMEDIATELY:
                                score, threshold = update_threshold(neuron_i, new_label_mask, threshold)
                            else:
                                neuron_masks = g["inspected_neurons"][neuron_i] > threshold
                                score = settings.SCORE_FUNCTION_OPTIMIZATION(neuron_masks, new_label_mask)
                        else:
                            score, threshold = settings.SCORE_FUNCTION_OPTIMIZATION(static_neuron_mask, new_label_mask), g["static_thresholds"][neuron_i]


                        # ------------------------------------------------------------------
                        # - Add a (as of now, tiny) complexity penalty for formula length. -
                        # ------------------------------------------------------------------
                        score = apply_complexity_penalty(score, len(new_formula))


                        # ---------------------------------------------------------------------
                        # - Add the new formula and its score and threshold to the dictionary -
                        # ---------------------------------------------------------------------
                        beam_with_new_formulas[new_formula] = (score, threshold)


                        # ----------------------------------
                        # - Update all tracking variables. -
                        # ----------------------------------
                        codes.append(new_formula.to_code(neuron_i=neuron_i))
                        formula_records.append((formula.to_str(), score))


                    # ----------------------------------------------------------------------------------
                    # - Delete all of the "close masks"; we don't have enough RAM to keep them around. -
                    # ----------------------------------------------------------------------------------
                    if close_masks is not None:
                        for close_formula in close_masks:
                            if close_formula is not None:
                                try:
                                    del known_masks[close_formula.to_code(neuron_i=neuron_i)]
                                except KeyError:
                                    # This can happen if another process deleted the same formula first. It's not a bug, so just do nothing.
                                    pass


        beam = initialize_beam(formulas=beam_with_new_formulas)
        print_beam()

    # ----------------------------------------------------------------------------------
    # - Now that beam search has concluded, collect the information we wish to return. -
    # ----------------------------------------------------------------------------------

    # Get the formula with highest score. Do not use counter.most_common() for this as that will throw an error if there is a tie,
    #   because once score and threshold are equal, it will attempt to compare two boolean numpy arrays, which will return a numpy boolean
    #   array rather than a boolean (hence 'the truth value of an array with more than one element is ambiguous'). This code is functionally
    #   identical except that (a) it only considers the score, and (b) in the case of a tie simply returns the first element.
    i = np.argmax([score for score, _ in beam.values()])
    w_formula, (_, w_threshold) = list(beam.items())[i]

    for code in stored_mask_codes:
        try:
            del known_masks[code]
        except KeyError:
            pass

    if settings.EASY_MODE:
        MaskLoader.delete_easy_masks(neuron_i)

    return neuron_i, w_formula, w_threshold, formula_records, history

def apply_complexity_penalty(iou, formula_length):
    """
    Compute a complexity penalty for formulas based on their length (to prevent long formulas that add no value). This is currently tiny.
    """
    return iou * (1 - settings.COMPLEXITY_PENALTY * (formula_length - 1))

def update_threshold(neuron_i, label_masks, threshold, direction='up'):
    """
    CURRENTLY NOT IN USE. Once upon a time, I thought it was a good idea to let the algorithm choose the threshold dynamically for each
    neuron. This turned out to be trickier than expected, although it is doable and comes with significant increases in performance. (To
    what degree that increase is valuable can be debated; I still think it has merit). The problem is that it massively increases runtime,
    making it completely unfeasible to be used in combination with CLOSE-TO masks. (No this can't be fixed with optimizations, the coded is
    already heavily optimized.) Since I won't get rid of CLOSE-TO masks, dynamic thresholding had to go instead.
    """

    def update_threshold_once(neuron_i, label_masks, threshold, direction):
        new_threshold = threshold + settings.THRESHOLD_STEP if direction == 'up' else threshold - settings.THRESHOLD_STEP
        new_neuron_masks = g["inspected_neurons"][neuron_i] > new_threshold

        new_iou = settings.SCORE_FUNCTION_OPTIMIZATION(new_neuron_masks, label_masks)

        return new_iou, new_threshold, new_neuron_masks

    neuron_masks = g["inspected_neurons"][neuron_i] > threshold

    iou = settings.SCORE_FUNCTION_OPTIMIZATION(neuron_masks, label_masks) # Override neuron Mask = True


    del neuron_masks

    # Begin by attempting one update step, then react based on whether or not it yielded an improvement
    new_iou, *rest = update_threshold_once(neuron_i, label_masks, threshold, direction)

    if new_iou <= iou: # no improvement; either try other direction, or (if we're already at 'down') just give up
        if direction == 'up':
            return update_threshold(neuron_i, label_masks, threshold, 'down')
        else:
            return new_iou, threshold


    # At this point, we know that there was nonzero improvement, so we can adapt the values in rest
    while new_iou - iou > settings.THRESHOLD_MIN_IMPROVEMENT:
        if random.random() < 0.0005:
            PI.show_simple(f"Random threshold update: {direction.upper()} with iou: {iou:.3f}, threshold: {threshold:.1f} (1 of 2000)")
        iou = new_iou
        threshold, neuron_masks = rest
        new_iou, *rest = update_threshold_once(neuron_i, label_masks, threshold, direction)

    # At this point, the loop exited, and the new threshold, mask, and hits may or may not be an improvement
    if new_iou > iou:
        threshold, neuron_masks = rest
        iou = new_iou

    return iou, threshold

def iou(neuron_mask, label_mask):
    """ Computes regular IoU score. """
    if settings.EASY_MODE:
        intersection = np.logical_and(neuron_mask[:len(label_mask)], label_mask).sum()
        union = np.logical_or(neuron_mask[:len(label_mask)], label_mask).sum()
    else:
        intersection = np.logical_and(neuron_mask, label_mask).sum()
        union = neuron_mask.sum() + label_mask.sum() - intersection

    return intersection / (union + 1e-10)

def imrou(map_im_2_neuron_mask, map_im_2_label_mask):
    """ Computes the not-normalized ImRoU_r score, with r = settings.MULTIPLIER_IP. """

    neuron_area_total, label_area_total, intersection_total, random_intersection_total, max_random_intersection_total = 0, 0, 0, 0, 0
    I = map_im_2_label_mask.shape[1] * map_im_2_label_mask.shape[2]
    r = settings.MULTIPLIER_IP

    for neuron_mask, label_mask in zip(map_im_2_neuron_mask, map_im_2_label_mask):
        neuron_area_image = neuron_mask.sum()
        label_area_image = label_mask.sum()

        neuron_area_total += neuron_area_image
        label_area_total += label_area_image
        intersection_total += np.logical_and(neuron_mask, label_mask).sum()
        random_intersection_total += neuron_area_image * label_area_image
        max_random_intersection_total += neuron_area_image * neuron_area_image

    union_total = neuron_area_total + label_area_total - intersection_total + 1e-10

    return (intersection_total - (r / I) * random_intersection_total) / union_total

def imr(map_im_2_neuron_mask, map_im_2_label_mask):
    """
    CURRENTLY NOT IN USE. A different metric I tried is to just subtract random intersection from actual intersection, without dividing by
    the union.
    """
    ipenalty_total = 0  # the expected size of the intersection between neuron and label if masks were chosen randomly but with the same size
    intersection_total = 0
    neuron_area_total = 0
    label_area_total = 0
    shape = map_im_2_label_mask.shape

    for neuron_mask, label_mask in zip(map_im_2_neuron_mask, map_im_2_label_mask):
        neuron_area_image = neuron_mask.sum()
        neuron_area_total += neuron_area_image

        label_area_image = label_mask.sum()
        label_area_total += label_area_image

        intersection_image = np.logical_and(neuron_mask, label_mask).sum()


        intersection_total += intersection_image
        ipenalty_image = (neuron_area_image * label_area_image) / (shape[1] * shape[2])
        ipenalty_total += ipenalty_image

    return intersection_total - settings.MULTIPLIER_IP * ipenalty_total

def imroun(map_im_2_neuron_mask, map_im_2_label_mask):
    """ Computes the normalized ImRoU_r score, with r = settings.MULTIPLIER_IP. """

    neuron_area_total, label_area_total, intersection_total, random_intersection_total, max_random_intersection_total = 0, 0, 0, 0, 0
    I = map_im_2_label_mask.shape[1] * map_im_2_label_mask.shape[2]
    r = settings.MULTIPLIER_IP

    for neuron_mask, label_mask in zip(map_im_2_neuron_mask, map_im_2_label_mask):
        neuron_area_image = neuron_mask.sum()
        label_area_image = label_mask.sum()

        neuron_area_total += neuron_area_image
        label_area_total += label_area_image
        intersection_total += np.logical_and(neuron_mask, label_mask).sum()
        random_intersection_total += neuron_area_image * label_area_image
        max_random_intersection_total += neuron_area_image * neuron_area_image

    union_total = neuron_area_total + label_area_total - intersection_total + 1e-10

    imrou_score = (intersection_total - (r/I) * random_intersection_total) / union_total
    max_imrou_score = (neuron_area_total - (r/I) * max_random_intersection_total) / neuron_area_total

    return imrou_score / max_imrou_score
